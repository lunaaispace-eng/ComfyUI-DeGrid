"""Find and load a Flux.2 VAE for the Forge Neo enhancement script.

Detection is by safetensors header, not filename: a Flux.2 VAE has a 32-channel
``decoder.conv_in`` and the ``bn.*`` latent-normalisation buffers. Only the
header is read (8 bytes + JSON), never the tensors, so listing a folder of
VAEs costs nothing.

Loading mirrors ``backend/loader.py``'s ``AutoencoderKLFlux2`` branch, plus the
one thing that branch gets for free from ``replace_state_dict`` and a standalone
loader has to do itself: the official file ships in *diffusers* key naming
(``decoder.up_blocks.0.resnets...``) while Forge's class uses ldm naming
(``decoder.up.0.block...``). ``load_state_dict`` then reports every key missing
and every key unexpected and the model runs on random weights without raising
(round trip 14 dB instead of 35). ``convert_vae_state_dict`` fixes that.

``backend`` is imported lazily inside ``get()`` so this module imports without
a Forge checkout (the offline tests exercise the header logic only).
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any

ARCH_FLUX2 = "flux2"

_header_cache: dict[str, tuple[tuple, str | None]] = {}  # path -> (stamp, arch)
_cache: dict[str, Any] = {}  # path -> VAE; a single entry, so switching files frees the previous one


def read_header(path: os.PathLike | str) -> dict:
    """The safetensors header (tensor name -> {dtype, shape, data_offsets}), or {}."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or n > 256 * 1024 * 1024:
                return {}
            return json.loads(f.read(n))
    except (OSError, ValueError, struct.error):
        return {}


def detect_arch(header: dict) -> str | None:
    """'flux2' for a Flux.2 VAE header (either key naming), else None."""
    conv_in = header.get("decoder.conv_in.weight")
    shape = conv_in.get("shape") if isinstance(conv_in, dict) else None
    if not shape or len(shape) != 4 or int(shape[1]) != 32:
        return None
    if not any(k.startswith("bn.") for k in header):
        return None
    return ARCH_FLUX2


def arch_of(path: os.PathLike | str) -> str | None:
    path = str(path)
    try:
        info = os.stat(path)
        stamp = (info.st_mtime_ns, info.st_size)
    except OSError:
        return None
    cached = _header_cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    arch = detect_arch(read_header(path)) if path.lower().endswith(".safetensors") else None
    _header_cache[path] = (stamp, arch)
    return arch


def candidates(vae_dict: dict) -> list[str]:
    """Names from a Forge ``sd_vae.vae_dict`` whose file is a Flux.2 VAE, sorted."""
    return sorted(name for name, path in vae_dict.items() if arch_of(path) == ARCH_FLUX2)


def _build(path: str):
    from transformers.modeling_utils import no_init_weights

    from backend import memory_management
    from backend.loader import HF
    from backend.nn.vae import AutoencoderKLFlux2
    from backend.operations import using_forge_operations
    from backend.patcher.vae import VAE
    from backend.state_dict import load_state_dict
    from backend.utils import load_torch_file

    state_dict = load_torch_file(path)
    if "decoder.up_blocks.0.resnets.0.norm1.weight" in state_dict:
        from modules_forge.packages.huggingface_guess.diffusers_convert import convert_vae_state_dict

        state_dict = convert_vae_state_dict(state_dict)

    config = AutoencoderKLFlux2.load_config(os.path.join(HF, "black-forest-labs", "FLUX.2-klein-9B", "vae"))
    if int(state_dict["decoder.conv_in.weight"].shape[0]) == 384:  # small decoder variant
        config["dch"] = 96

    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype(), extra_dtype="vae"):
            model = AutoencoderKLFlux2.from_config(config)
    load_state_dict(model, state_dict, ignore_start="loss.")
    del state_dict
    return VAE(model=model, is_flux2=True)


def get(path: os.PathLike | str):
    """Load (or return the already loaded) Flux.2 VAE for ``path``."""
    path = str(path)
    if path not in _cache:
        from backend import memory_management

        vae = _build(path)
        for stale in _cache.values():
            try:
                memory_management.unload_model(stale.patcher)
            except Exception:
                pass
        _cache.clear()
        memory_management.soft_empty_cache()
        _cache[path] = vae
        memory_management.logger.info(f"VAE Enhance: loaded {os.path.basename(path)} (Flux.2 VAE)")
    return _cache[path]


def loaded_paths() -> list[str]:
    return list(_cache)
