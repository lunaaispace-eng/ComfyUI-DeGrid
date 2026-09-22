"""VAE DeGrid for Forge Neo (sd-webui-forge-classic, neo branch).

Removes the 2px pixel grid the Qwen Image / Wan 2.1 VAEs leave on decoded
images (Krea 2, Qwen Image, Wan, Anima). Same maths as the ComfyUI node in
this repo — ``degrid_core.py`` at the extension root is loaded as-is.

Integration: the checkpoint's VAE is replaced, for the duration of each
sampling pass, by a wrapper that runs the notch inside every decode
(see ``lib_degrid/vae_wrapper.py``). That means the hires-fix first pass is
degridded *before* the upscaler sees it, which an image-postprocessing hook
could not do, and the final decode is degridded before the uint8 conversion.

Hooks used:
- ``process_before_every_sampling`` — swap the VAE (fires per pass; Forge resets
  ``forge_objects`` right before it, so the swap is scoped to that pass).
- ``postprocess_batch`` — fires right after the final decode of a batch: log
  each decode, write the measurement to the infotext, collect previews.
- ``postprocess`` — undo the swap, append preview images to the results.
"""

from __future__ import annotations

import inspect

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import scripts
from modules.infotext_utils import PasteField
from modules.processing import logger
from modules.ui_components import InputAccordion

from lib_degrid.loader import load_core
from lib_degrid.vae_wrapper import DecodeRecord, DeGridConfig, restore, wrap

_EXTENSION_ROOT = scripts.basedir()

INFOTEXT_KEY = "DeGrid"
INFOTEXT_RESULT_KEY = "DeGrid result"
MODES = ("auto", "manual")
ZOOMS = ("4x", "8x")
DEFAULT_ZOOM = "8x"


def _ascii(text: str) -> str:
    """Console code pages on Windows choke on the status line's typography."""
    return text.replace("—", "-").replace("·", "|")


def _to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr)


def _paste(name: str, cast):
    """PasteField callable: pull one value out of the ``DeGrid`` infotext entry."""

    def read(params: dict):
        parsed = DeGridConfig.parse_infotext(params.get(INFOTEXT_KEY))
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


class DeGridScript(scripts.Script):
    # Runs before other VAE-swapping extensions (VAE Utils uses 15) so that they
    # see, and can replace, our wrapper rather than the other way round.
    sorting_priority = 10

    _records: list = []  # DecodeRecords since the last postprocess_batch (class-level: see knowledge.md #4)
    _previews: list = []  # (PIL, label) collected over the whole run
    _config: DeGridConfig | None = None

    def title(self):
        return "VAE DeGrid"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        core = load_core(_EXTENSION_ROOT)
        default_threshold = round(float(core.NEGLIGIBLE_AMP) * 255.0, 2)

        with InputAccordion(False, label=self.title()) as enabled:
            gr.HTML(
                "Removes the 2px pixel grid left by the Qwen Image / Wan 2.1 VAEs "
                "(Krea 2, Qwen Image, Wan, Anima). Runs inside the VAE decode, so the "
                "hires-fix first pass is cleaned before the upscaler sees it. "
                "Images without a grid pass through untouched."
            )
            with gr.Row():
                mode = gr.Radio(
                    choices=list(MODES), value="auto", label="Mode",
                    info="auto measures each image and sets the removal limit itself; manual uses the slider",
                )
                limit = gr.Slider(
                    minimum=0.0, maximum=0.10, step=0.001, value=0.02, label="Limit (manual mode)",
                    info="max per-pixel correction, 0-1 scale; the VAE grid is usually 0.005-0.02",
                )
            with gr.Row():
                skip_when_clean = gr.Checkbox(
                    value=True, label="Skip when clean",
                    info="pass an image through bit-for-bit when no lattice is measured",
                )
                threshold = gr.Slider(
                    minimum=0.1, maximum=3.0, step=0.05, value=default_threshold, label="Clean threshold (/255)",
                    info="lattice amplitude below which an image counts as clean; Krea 2 decodes measure 0.5-2.5",
                )
            with gr.Row():
                preview = gr.Checkbox(
                    value=False, label="Show removed grid",
                    info="adds a magnified centre crop of what was subtracted to the results",
                )
                preview_zoom = gr.Radio(choices=list(ZOOMS), value=DEFAULT_ZOOM, label="Preview zoom")
                grid_gain = gr.Slider(
                    minimum=1.0, maximum=50.0, step=1.0, value=10.0, label="Preview gain",
                    info="brightness of the preview only; never affects the image",
                )

        self.infotext_fields = [
            PasteField(enabled, _paste_enabled),
            PasteField(mode, _paste("mode", str)),
            PasteField(limit, _paste("limit", float)),
            PasteField(skip_when_clean, _paste("skip", _paste_bool)),
            PasteField(threshold, _paste("threshold", float)),
        ]

        # Positional order == process_before_every_sampling's parameter order.
        return [enabled, mode, limit, skip_when_clean, threshold, preview, preview_zoom, grid_gain]

    # -- per pass -------------------------------------------------------------

    @staticmethod
    def build_config(mode, limit, skip_when_clean, threshold, preview, preview_zoom, grid_gain) -> DeGridConfig:
        mode = str(mode) if str(mode) in MODES else "auto"
        try:
            zoom = int(str(preview_zoom).rstrip("xX"))
        except ValueError:
            zoom = int(DEFAULT_ZOOM.rstrip("x"))
        return DeGridConfig(
            mode=mode,
            limit=float(limit),
            skip_when_clean=bool(skip_when_clean),
            threshold_255=float(threshold),
            preview=bool(preview),
            preview_zoom=zoom,
            grid_gain=float(grid_gain),
        )

    def process_before_every_sampling(
        self,
        p,
        enabled,
        mode,
        limit,
        skip_when_clean,
        threshold,
        preview,
        preview_zoom,
        grid_gain,
        **kwargs,
    ):
        sd_model = getattr(p, "sd_model", None)
        if sd_model is None:
            return
        restore(sd_model)  # also self-heals after an interrupted previous run

        cls = DeGridScript
        if not enabled:
            cls._config = None
            return

        objects = getattr(sd_model, "forge_objects", None)
        source = getattr(objects, "vae", None) if objects is not None else None
        if source is None:
            logger.warning("DeGrid: the loaded model has no VAE object to wrap; skipping")
            cls._config = None
            return

        config = self.build_config(mode, limit, skip_when_clean, threshold, preview, preview_zoom, grid_gain)
        core = load_core(_EXTENSION_ROOT)
        objects.vae = wrap(source, config, core, cls._records.append)
        cls._config = config
        p.extra_generation_params[INFOTEXT_KEY] = config.infotext()

    # -- per batch, right after the final decode --------------------------------

    def postprocess_batch(self, p, *args, **kwargs):
        cls = DeGridScript
        config = cls._config
        records: list[DecodeRecord] = list(cls._records)
        cls._records.clear()
        if config is None or not records:
            return

        core = load_core(_EXTENSION_ROOT)
        for record in records:
            for i, stats in enumerate(record.stats):
                line = core.status_line(config.mode, [stats], threshold=config.threshold)
                index = f" [{i + 1}/{record.count}]" if record.count > 1 else ""
                logger.info(_ascii(f"DeGrid: {record.width}x{record.height}{index} | {line}"))

        final = records[-1]  # the last decode of the batch is the one that became the images
        result = core.status_line(config.mode, final.stats, threshold=config.threshold)
        p.extra_generation_params[INFOTEXT_RESULT_KEY] = _ascii(result)

        if config.preview:
            for i, vis in enumerate(final.previews):
                if vis is None:
                    continue
                label = _ascii(core.status_line(config.mode, [final.stats[i]], threshold=config.threshold))
                cls._previews.append((_to_pil(vis), f"DeGrid removed grid ({config.preview_zoom}x zoom, gain {config.grid_gain:g}) | {label}"))

    # -- end of run ---------------------------------------------------------------

    def postprocess(self, p, processed, *args):
        cls = DeGridScript
        restore(getattr(p, "sd_model", None))

        previews, cls._previews = list(cls._previews), []
        cls._records.clear()
        cls._config = None

        extra_images = getattr(processed, "extra_images", None)
        infotexts = getattr(processed, "infotexts", None)
        if extra_images is None:
            return
        # The preview's infotext is the run's own: adding a line after the
        # parameters line would break "send to txt2img" parsing for that image.
        base_info = infotexts[0] if infotexts else ""
        for image, _label in previews:
            extra_images.append(image)
            if infotexts is not None:
                infotexts.append(base_info)


# Sanity check kept next to the code it protects: ui() and the hook must agree.
_HOOK_PARAMS = [
    name
    for name in inspect.signature(DeGridScript.process_before_every_sampling).parameters
    if name not in ("self", "p", "kwargs")
]
assert _HOOK_PARAMS == ["enabled", "mode", "limit", "skip_when_clean", "threshold", "preview", "preview_zoom", "grid_gain"], _HOOK_PARAMS
