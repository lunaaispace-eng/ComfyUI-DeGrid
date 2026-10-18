"""Fake Forge Neo ``modules``/``gradio`` pieces for the offline tests.

Not a test file. The real script file is loaded the way Forge's own
``modules.script_loading.load_module()`` loads it (``spec_from_file_location``)
against these fakes, so the hook wiring is exercised without a Forge checkout,
a checkpoint, or gradio.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


class Component:
    def __init__(self, *args, **kwargs):
        self.value = kwargs.get("value")
        self.label = kwargs.get("label")
        self.choices = kwargs.get("choices")
        self.clicks = []

    def click(self, fn=None, **kwargs):
        self.clicks.append((fn, kwargs))


class Container(Component):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def fake_gradio() -> types.ModuleType:
    gr = types.ModuleType("gradio")
    for name in ("HTML", "Markdown", "Checkbox", "Slider", "Radio", "Dropdown", "Button"):
        setattr(gr, name, Component)
    for name in ("Row", "Column", "Group", "Accordion"):
        setattr(gr, name, Container)
    gr.update = lambda **kwargs: dict(kwargs)
    return gr


def fake_forge_modules(extension_root: str) -> dict[str, types.ModuleType]:
    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []

    scripts_mod = types.ModuleType("modules.scripts")
    scripts_mod.basedir = lambda: extension_root

    class Script:
        sorting_priority = 0

        def show(self, is_img2img):
            return None

        def ui(self, is_img2img):
            return []

        def process(self, p, *args, **kwargs):
            pass

        def process_before_every_sampling(self, p, *args, **kwargs):
            pass

        def postprocess_batch(self, p, *args, **kwargs):
            pass

        def postprocess_image_after_composite(self, p, pp, *args):
            pass

        def postprocess(self, p, processed, *args):
            pass

    class PostprocessImageArgs:
        def __init__(self, image, index):
            self.image = image
            self.index = index

    scripts_mod.Script = Script
    scripts_mod.AlwaysVisible = object()
    scripts_mod.PostprocessImageArgs = PostprocessImageArgs

    ui_components_mod = types.ModuleType("modules.ui_components")

    class InputAccordion(Container):
        def __init__(self, value=False, label=None, **kwargs):
            super().__init__(value=value, label=label)

    ui_components_mod.InputAccordion = InputAccordion
    ui_components_mod.ToolButton = Component

    ui_mod = types.ModuleType("modules.ui")
    ui_mod.refresh_symbol = "R"

    sd_vae_mod = types.ModuleType("modules.sd_vae")
    sd_vae_mod.vae_dict = {}
    sd_vae_mod.refresh_calls = 0

    def refresh_vae_list():
        sd_vae_mod.refresh_calls += 1

    sd_vae_mod.refresh_vae_list = refresh_vae_list

    infotext_utils_mod = types.ModuleType("modules.infotext_utils")

    class PasteField(tuple):
        def __new__(cls, component, target, *, api=None):
            return super().__new__(cls, (component, target))

        def __init__(self, component, target, *, api=None):
            self.component = component
            self.function = target if callable(target) else None
            self.label = target if isinstance(target, str) else None

    infotext_utils_mod.PasteField = PasteField

    processing_mod = types.ModuleType("modules.processing")
    processing_mod.logger = logging.getLogger("processing")

    class StableDiffusionProcessing:
        pass

    processing_mod.StableDiffusionProcessing = StableDiffusionProcessing

    modules_pkg.scripts = scripts_mod
    modules_pkg.ui_components = ui_components_mod
    modules_pkg.infotext_utils = infotext_utils_mod
    modules_pkg.processing = processing_mod
    modules_pkg.ui = ui_mod
    modules_pkg.sd_vae = sd_vae_mod
    return {
        "modules": modules_pkg,
        "modules.scripts": scripts_mod,
        "modules.ui_components": ui_components_mod,
        "modules.infotext_utils": infotext_utils_mod,
        "modules.processing": processing_mod,
        "modules.ui": ui_mod,
        "modules.sd_vae": sd_vae_mod,
    }


def install_fakes(extension_root: str = str(REPO_ROOT)):
    sys.modules["gradio"] = fake_gradio()
    sys.modules.update(fake_forge_modules(extension_root))
    if extension_root not in sys.path:
        sys.path.insert(0, extension_root)


def load_script_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- a stand-in for backend.patcher.vae.VAE with the same decode topology -------


class FakeDecoder:
    """Pretends to be ``first_stage_model``: returns a fixed image in -1..1,
    channels-first, ignoring the latent except for its batch size."""

    def __init__(self, image01: torch.Tensor):
        self.image01 = image01  # [1, C, (T,) H, W]

    def decode(self, samples):
        b = samples.shape[0]
        return self.image01.expand(b, *self.image01.shape[1:]).mul(2.0).sub(1.0)


class FakeVAE:
    """Mirrors ``backend/patcher/vae.py::VAE``'s call graph: ``decode`` runs
    ``process_output`` once per sub-batch and falls back to ``decode_tiled``
    (which calls ``process_output`` once) on OOM."""

    def __init__(self, decoder: FakeDecoder, is_wan: bool = True, device: str = "cpu"):
        self.first_stage_model = decoder
        self.patcher = object()
        self.device = torch.device(device)
        self.output_device = torch.device("cpu")
        self.is_wan = is_wan
        self.latent_channels = 16
        self.oom_once = False
        self.tiled_calls = 0
        self.encode_calls = 0

    def decode(self, samples_in):
        if self.oom_once:
            self.oom_once = False
            return self.decode_tiled(samples_in)
        outs = []
        for i in range(samples_in.shape[0]):
            raw = self.first_stage_model.decode(samples_in[i : i + 1]).to(torch.float32).clone()
            outs.append(self.process_output(raw))
        return torch.cat(outs, dim=0).movedim(1, -1)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        self.tiled_calls += 1
        raw = self.first_stage_model.decode(samples).to(torch.float32).clone()
        return self.process_output(raw).movedim(1, -1)

    def encode(self, pixel_samples):
        self.encode_calls += 1
        return torch.zeros(pixel_samples.shape[0], self.latent_channels, 1, 8, 8)

    @staticmethod
    def process_output(image):
        return image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)


class FakeObjects:
    def __init__(self, vae):
        self.unet = object()
        self.clip = object()
        self.vae = vae
        self.clipvision = None

    def shallow_copy(self):
        return FakeObjects(self.vae)


class FakeModel:
    def __init__(self, vae):
        self.forge_objects_original = FakeObjects(vae)
        self.forge_objects_after_applying_lora = FakeObjects(vae)
        self.forge_objects = FakeObjects(vae)
        self.is_wan = getattr(vae, "is_wan", True)


class FakeProcessing:
    def __init__(self, vae):
        self.sd_model = FakeModel(vae)
        self.extra_generation_params = {}


class FakeProcessed:
    def __init__(self):
        self.images = []
        self.extra_images = []
        self.infotexts = ["prompt\nSteps: 1"]


class FakeFlux2VAE:
    """An *identity* codec with the Flux.2 VAE's interface and geometry: 16px
    stride, [B, H, W, C] in and out. encode = pixel-unshuffle by 16 (so the
    latent grid is H/16 x W/16), decode = pixel-shuffle back. Any latent-space
    extrapolation therefore decodes to exactly the same operation in pixel
    space, which makes the enhancement maths testable to float precision."""

    latent_channels = 128  # what describe_codec() looks at; the fake's real channel count is 768
    downscale_ratio = 16

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.encode_calls = 0
        self.decode_calls = 0
        self.first_stage_model = None

    def encode(self, pixels):  # [B, H, W, C] -> [B, C*256, H/16, W/16]
        self.encode_calls += 1
        x = pixels.movedim(-1, 1).float()
        assert x.shape[-2] % 16 == 0 and x.shape[-1] % 16 == 0, tuple(x.shape)
        return torch.nn.functional.pixel_unshuffle(x, 16)

    def decode(self, latent):
        self.decode_calls += 1
        return torch.nn.functional.pixel_shuffle(latent.float(), 16).movedim(1, -1)


# -- synthetic images ------------------------------------------------------------


def smooth_image(h: int, w: int, c: int = 3, seed: int = 0) -> torch.Tensor:
    """A soft gradient with a few blurred blobs: no lattice, some real detail. [1, c, h, w]"""
    g = torch.Generator().manual_seed(seed)
    yy = torch.linspace(0.2, 0.8, h).view(h, 1)
    xx = torch.linspace(0.3, 0.7, w).view(1, w)
    base = (yy * 0.6 + xx * 0.4).expand(c, h, w).clone()
    noise = torch.rand(1, c, h // 8, w // 8, generator=g)
    noise = torch.nn.functional.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)[0]
    return (base * 0.7 + noise * 0.3).clamp(0.0, 1.0).unsqueeze(0)


def add_lattice(img: torch.Tensor, v_amp_255: float = 1.5, h_amp_255: float = 1.0) -> torch.Tensor:
    """Add phase-locked 2px vertical and horizontal stripes, amplitudes in /255 (peak)."""
    h, w = img.shape[-2:]
    sx = ((torch.arange(w) % 2) * 2 - 1).float().view(1, 1, 1, w)
    sy = ((torch.arange(h) % 2) * 2 - 1).float().view(1, 1, h, 1)
    return (img + sx * (v_amp_255 / 255.0) + sy * (h_amp_255 / 255.0)).clamp(0.0, 1.0)
