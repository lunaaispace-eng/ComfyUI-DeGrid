"""VAE Enhance for Forge Neo (sd-webui-forge-classic, neo branch).

Detail extrapolation through a second VAE (Flux.2) on the finished image:
encode, encode a blurred copy, push the latent away from the blurred one,
decode. Same maths as the ComfyUI node in this repo — ``vae_enhance_core.py``
at the extension root is loaded as-is; the notch that runs first is
``degrid_core.py``.

Integration: the image-postprocessing family, ``postprocess_image_after_composite``,
which fires once per image after the *final* decode. That is deliberate: the
notch (the VAE DeGrid script in this same extension) has to run on every
decode so the hires upscaler never sees a lattice, but the extrapolation must
run once, on the final pixels, so its fidelity cost is paid once and no
upscaler is fed re-drawn strands.

The notch is also run here on the input, in auto mode, whether or not the
DeGrid accordion is enabled: blurring removes the 2px grid, so the detail
direction would otherwise carry the lattice and the extrapolation would draw
it back. On an already-notched decode that pass measures the image as clean
and touches nothing.
"""

from __future__ import annotations

import inspect
import os
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import scripts, sd_vae
from modules.infotext_utils import PasteField
from modules.processing import logger
from modules.ui import refresh_symbol
from modules.ui_components import InputAccordion, ToolButton

from lib_degrid import flux2_vae
from lib_degrid.loader import load_core, load_enhance_core

_EXTENSION_ROOT = scripts.basedir()

INFOTEXT_KEY = "VAE Enhance"
INFOTEXT_RESULT_KEY = "VAE Enhance result"
NONE = "None"

_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


def _ascii(text: str) -> str:
    return text.replace("—", "-").replace("·", "|").replace("→", "->")


def _to_tensor(image: Image.Image) -> tuple[torch.Tensor, str]:
    mode = image.mode if image.mode in ("RGB", "RGBA") else "RGB"
    if image.mode != mode:
        image = image.convert(mode)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0), mode


def _to_image(t: torch.Tensor, mode: str = "RGB") -> Image.Image:
    arr = t.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
        mode = "L"
    return Image.fromarray(arr, mode=mode)


def _paste(name: str, cast):
    def read(params: dict):
        parsed = parse_infotext(params.get(INFOTEXT_KEY))
        if not parsed or name not in parsed:
            return None
        try:
            return cast(parsed[name])
        except (TypeError, ValueError):
            return None

    return read


def _paste_enabled(params: dict):
    return INFOTEXT_KEY in params


def _paste_bool(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def infotext(vae_name, gain, sigma, floor, target, skin_only, tone_fix, post_notch) -> str:
    """Compact ``k=v;k=v`` form; no colons so Forge writes it unquoted."""
    return ";".join(
        [
            f"vae={vae_name}",
            f"gain={float(gain):g}",
            f"sigma={float(sigma):g}",
            f"floor={float(floor):g}",
            f"target={float(target):g}",
            f"skin={int(bool(skin_only))}",
            f"tonefix={int(bool(tone_fix))}",
            f"postnotch={int(bool(post_notch))}",
        ]
    )


def parse_infotext(text: str | None) -> dict[str, str] | None:
    if not text or not isinstance(text, str):
        return None
    out: dict[str, str] = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out or None


def vae_choices() -> list[str]:
    names = flux2_vae.candidates(sd_vae.vae_dict)
    return names if names else [NONE]


def resolve_vae_path(vae_name: str) -> str | None:
    if vae_name in (None, "", NONE):
        return None
    path = sd_vae.vae_dict.get(vae_name)
    if path is None:
        sd_vae.refresh_vae_list()
        path = sd_vae.vae_dict.get(vae_name)
    return None if path is None else str(path)


class VAEEnhanceScript(scripts.Script):
    sorting_priority = 20  # after VAE DeGrid (10); irrelevant to the decode hook, keeps the accordions in a sensible order

    _previews: list = []  # (PIL, label) collected over the whole run
    _warned: set = set()  # one warning per (reason) per run

    def title(self):
        return "VAE Enhance (Flux.2 round trip)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        enhance_core = load_enhance_core(_EXTENSION_ROOT)
        choices = vae_choices()

        with InputAccordion(False, label=self.title()) as enabled:
            gr.HTML(
                "Re-draws fine texture (skin pores, fur, hair) by pushing the image through a "
                "<b>Flux.2 VAE</b>: encode, encode a blurred copy, extrapolate away from the "
                "blurred latent, decode. Runs once on each final image. The 2px VAE grid is "
                "removed first (auto notch) whether or not VAE DeGrid is enabled. Needs a "
                "Flux.2 VAE file in <code>models/VAE</code> (detected by its header, not its name)."
            )
            with gr.Row():
                vae_name = gr.Dropdown(
                    value=choices[0], choices=choices, label="Flux.2 VAE",
                    info="only Flux.2 VAEs are listed; put flux2-vae.safetensors in models/VAE and refresh",
                )
                refresh = ToolButton(value=refresh_symbol, tooltip="Refresh the list of VAE files")
            with gr.Row():
                gain = gr.Slider(
                    minimum=0.0, maximum=2.0, step=0.05, value=float(enhance_core.GAIN), label="Gain",
                    info="0.5 is the safe default; skin and fur improve from 0.5, regular fabric weave moires from 0.75",
                )
                sigma = gr.Slider(
                    minimum=0.5, maximum=2.0, step=0.1, value=float(enhance_core.SIGMA), label="Blur sigma (px)",
                    info="scale of the detail direction; 1 px is right for pores and fur, 2 pushes structure",
                )
            with gr.Row():
                floor = gr.Slider(
                    minimum=0.0, maximum=10.0, step=0.5, value=float(enhance_core.FLOOR), label="Mask floor (/255)",
                    info="regions with less local texture get nothing; clean sky 0.35, sand 1.4, flat wall 1.5, plastic skin 1.2-2: 0.5 only drops sky, 2.5-3.5 protects walls but loses plastic skin; 0 = off",
                )
                target = gr.Slider(
                    minimum=0.0, maximum=20.0, step=0.5, value=float(enhance_core.TARGET), label="Texture target (/255)",
                    info="gain fades out as a region approaches this texture level; 0 = off (use off for animals, fur is above it)",
                )
            with gr.Row():
                skin_only = gr.Checkbox(
                    value=True, label="Skin tones only",
                    info="apply the gain only where the colour is skin-like (also brown hair, beige fur); off for animals and greyscale",
                )
                tone_fix = gr.Checkbox(
                    value=True, label="Keep input colour",
                    info="the decoder shifts hue and shading with the detail; keep the input's 17px low-pass",
                )
                post_notch = gr.Checkbox(
                    value=False, label="Notch again after",
                    info="the Flux.2 decoder has a faint 2px checker of its own that grows with gain; costs 2px texture",
                )
                show_mask = gr.Checkbox(
                    value=False, label="Show mask",
                    info="adds the per-latent-pixel weight map to the results",
                )

        def do_refresh():
            sd_vae.refresh_vae_list()
            new = vae_choices()
            return gr.update(choices=new, value=new[0])

        refresh.click(fn=do_refresh, outputs=[vae_name], show_progress=False)

        self.infotext_fields = [
            PasteField(enabled, _paste_enabled),
            PasteField(vae_name, _paste("vae", str)),
            PasteField(gain, _paste("gain", float)),
            PasteField(sigma, _paste("sigma", float)),
            PasteField(floor, _paste("floor", float)),
            PasteField(target, _paste("target", float)),
            PasteField(skin_only, _paste("skin", _paste_bool)),
            PasteField(tone_fix, _paste("tonefix", _paste_bool)),
            PasteField(post_notch, _paste("postnotch", _paste_bool)),
        ]

        # Positional order == the hooks' parameter order.
        return [enabled, vae_name, gain, sigma, floor, target, skin_only, tone_fix, post_notch, show_mask]

    # -- once per run -------------------------------------------------------------

    def process(self, p, enabled, vae_name, gain, sigma, floor, target, skin_only, tone_fix, post_notch, show_mask, **kwargs):
        cls = VAEEnhanceScript
        cls._previews = []
        cls._warned = set()
        if not enabled:
            return
        if resolve_vae_path(vae_name) is None:
            logger.error(f'VAE Enhance: no Flux.2 VAE named "{vae_name}"; the images will pass through unchanged')
            cls._warned.add("no-vae")
            return
        p.extra_generation_params[INFOTEXT_KEY] = infotext(vae_name, gain, sigma, floor, target, skin_only, tone_fix, post_notch)

    # -- once per image, after the final decode -------------------------------------

    @torch.inference_mode()
    def postprocess_image_after_composite(self, p, pp, enabled, vae_name, gain, sigma, floor, target, skin_only, tone_fix, post_notch, show_mask, **kwargs):
        cls = VAEEnhanceScript
        if not enabled or "no-vae" in cls._warned:
            return
        image = getattr(pp, "image", None)
        if not isinstance(image, Image.Image):
            return
        path = resolve_vae_path(vae_name)
        if path is None:
            return

        try:
            vae = flux2_vae.get(path)
        except Exception as e:  # a broken file must not kill the run
            if "load" not in cls._warned:
                logger.error(f"VAE Enhance: failed to load {os.path.basename(path)}\n{e}")
                cls._warned.add("load")
            return

        degrid_core = load_core(_EXTENSION_ROOT)
        enhance_core = load_enhance_core(_EXTENSION_ROOT)
        x, mode = _to_tensor(image)
        work = getattr(vae, "device", None)
        work = work if isinstance(work, torch.device) else torch.device("cpu")

        t0 = time.perf_counter()
        try:
            out, mask, line = self._run(x.to(work), vae, degrid_core, enhance_core, gain, sigma, floor, target, skin_only, tone_fix, post_notch)
        except _OOM:
            torch.cuda.empty_cache()
            logger.error("VAE Enhance: out of memory; this image is left unchanged")
            return
        seconds = time.perf_counter() - t0
        line = f"{line} · {seconds:.1f} s"

        index = getattr(pp, "index", 0)
        logger.info(_ascii(f"VAE Enhance: {image.width}x{image.height} [{int(index) + 1}] | {line}"))
        p.extra_generation_params[INFOTEXT_RESULT_KEY] = _ascii(line)
        pp.image = _to_image(out, mode)
        if show_mask:
            cls._previews.append((_to_image(mask, "RGB"), _ascii(f"VAE Enhance mask | {line}")))

    @staticmethod
    def _run(x, vae, degrid_core, enhance_core, gain, sigma, floor, target, skin_only, tone_fix, post_notch):
        """The whole per-image pipeline on a [1, H, W, C] 0..1 tensor; returns (out, mask, status line)."""
        cleaned, _, dg = degrid_core.degrid(x, mode="auto")
        out, mask, st = enhance_core.enhance(
            cleaned, vae,
            sigma=float(sigma), gain=float(gain), floor=float(floor), target=float(target),
            skin_only=bool(skin_only), tone=bool(tone_fix), work_device=x.device,
        )
        if post_notch:
            out, _, _ = degrid_core.degrid(out, mode="auto")
        grid_out = degrid_core.lattice_amp(degrid_core.extract_grid(out[..., :3].permute(0, 3, 1, 2).float()))[0][0].item() * 255.0
        line = enhance_core.status_line(
            st,
            grid_in_255=dg[0]["amp_255"], grid_in_removed=not dg[0]["skipped"], grid_out_255=grid_out,
        )
        name, ok = enhance_core.describe_codec(vae)
        if not ok:
            line = f"untested VAE {name} · {line}"
        return out, mask, line

    # -- end of run ---------------------------------------------------------------

    def postprocess(self, p, processed, *args):
        cls = VAEEnhanceScript
        previews, cls._previews = list(cls._previews), []
        cls._warned = set()

        extra_images = getattr(processed, "extra_images", None)
        infotexts = getattr(processed, "infotexts", None)
        if extra_images is None:
            return
        base_info = infotexts[0] if infotexts else ""
        for image, _label in previews:
            extra_images.append(image)
            if infotexts is not None:
                infotexts.append(base_info)


_UI_PARAMS = ["enabled", "vae_name", "gain", "sigma", "floor", "target", "skin_only", "tone_fix", "post_notch", "show_mask"]
for _hook in (VAEEnhanceScript.process, VAEEnhanceScript.postprocess_image_after_composite):
    _params = [n for n in inspect.signature(_hook).parameters if n not in ("self", "p", "pp", "kwargs")]
    assert _params == _UI_PARAMS, (_hook.__name__, _params)
