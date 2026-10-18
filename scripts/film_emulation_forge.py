"""Film emulation for Forge Neo (sd-webui-forge-classic, neo branch).

Optional, off by default, the last stage of the pipeline. Its stage today is
film grain: a seeded, spatially correlated noise field on each final image,
coarser and stronger in the shadows. Same maths as the ComfyUI node in this
repo; ``film_emulation_core.py`` at the extension root is loaded as-is.

Integration: ``postprocess_image_after_composite``, once per final image, with
``sorting_priority = 30`` so it runs after VAE DeGrid (10) and VAE Enhance (20).
It never touches the hires-fix first pass: grain added before a resize or an
upscaler is resampled into blobs, so it has to be the last thing that happens
to the pixels. It does not depend on either other accordion being enabled.

The seed is the image's own generation seed (``p.all_seeds[index]``), so a batch
gets different grain per image and a re-run gives the same grain.

Images made before the rename carry ``Film grain: iso=...;strength=...``; the
paste fields read that too, so they restore into this accordion.
"""

from __future__ import annotations

import inspect
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import scripts
from modules.infotext_utils import PasteField
from modules.processing import logger
from modules.ui_components import InputAccordion

from lib_degrid.loader import load_film_core

_EXTENSION_ROOT = scripts.basedir()

INFOTEXT_KEY = "Film emulation"
INFOTEXT_RESULT_KEY = "Film emulation result"
LEGACY_INFOTEXT_KEY = "Film grain"  # before the rename; "strength" there is "grain" here
_LEGACY_NAMES = {"grain": "strength"}


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
    return Image.fromarray(arr, mode=mode)


OPTIC_KEYS = (("softness", "soft"), ("halation", "halation"), ("bloom", "bloom"), ("highlight_rolloff", "rolloff"))


def infotext(iso, grain, colour_grain, match_texture, film_type, softness=0, halation=0, bloom=0, highlight_rolloff=0) -> str:
    """Compact ``k=v;k=v`` form; no colons or commas so Forge writes it unquoted."""
    optics = "".join(f";{k}={float(v):g}" for (_, k), v in zip(OPTIC_KEYS, (softness, halation, bloom, highlight_rolloff)))
    return f"iso={str(iso).replace(' ', '')};grain={float(grain):g};chroma={colour_grain};match={int(bool(match_texture))};film={film_type}{optics}"


def parse_infotext(text: str | None) -> dict[str, str] | None:
    if not text or not isinstance(text, str):
        return None
    out: dict[str, str] = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out or None


def _paste(name: str, cast):
    def read(params: dict):
        parsed = parse_infotext(params.get(INFOTEXT_KEY))
        key = name
        if parsed is None:
            parsed = parse_infotext(params.get(LEGACY_INFOTEXT_KEY))
            key = _LEGACY_NAMES.get(name, name)
        if not parsed or key not in parsed:
            return None
        try:
            return cast(parsed[key])
        except (TypeError, ValueError):
            return None

    return read


def _paste_optic(name: str):
    """An optical stage missing from a written infotext was off (older images): 0, not 'keep'."""
    read = _paste(name, float)

    def optic(params: dict):
        v = read(params)
        if v is None and _paste_enabled(params):
            return 0.0
        return v

    return optic


def _paste_enabled(params: dict):
    return INFOTEXT_KEY in params or LEGACY_INFOTEXT_KEY in params


def _paste_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _paste_iso(v: str) -> str:
    """'ISO200' (as written) or 'ISO 200' back to the dropdown's label."""
    digits = "".join(ch for ch in str(v) if ch.isdigit())
    return f"ISO {digits}"


def seed_for(p, index: int) -> int:
    seeds = getattr(p, "all_seeds", None)
    if isinstance(seeds, (list, tuple)) and seeds and 0 <= index < len(seeds):
        try:
            return int(seeds[index])
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(p, "seed", 0)) + int(index)
    except (TypeError, ValueError):
        return int(index)


class FilmEmulationScript(scripts.Script):
    sorting_priority = 30  # after VAE DeGrid (10) and VAE Enhance (20): film emulation is the last thing that touches the pixels

    def title(self):
        return "Film emulation"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        core = load_film_core(_EXTENSION_ROOT)
        with InputAccordion(False, label=self.title()) as enabled:
            gr.HTML(
                "Makes each final image look shot on film, as the last step. <b>Grain</b>: "
                "<b>ISO</b> sets its character (blob size, clumpiness, colour), never its amount; "
                "ISO 400 matches the grain the model draws itself when prompted for it. The "
                "<b>Grain</b> slider sets the amount alone: 1 is invisible, 50 reads as texture "
                "rather than noise, 100 is a bit noisy, the same on every ISO. Grain is coarser and "
                "stronger in the shadows; white photo borders get none. Grey stays exactly grey, "
                "colour grain never adds a hue and is off on monochrome images. Judge the result at 100%."
            )
            with gr.Row():
                iso = gr.Dropdown(
                    value=core.ISO_DEFAULT, choices=list(core.ISO_NAMES), label="ISO",
                    info="grain character, not amount, named after real stock; 100-200 blend into skin as fine texture, 400 is the model's own grain, 800+ sits on the image as film",
                )
                grain = gr.Slider(
                    minimum=0, maximum=100, step=1, value=int(core.STRENGTH_DEFAULT), label="Grain",
                    info="0 = off, 1 = 0.3/255 invisible, 50 = 3.3/255 texture not noise, 100 = 8/255 grainy (midtones; the film type sets the other tones)",
                )
                colour_grain = gr.Radio(
                    choices=list(core.CHROMA_MODES), value="auto", label="Colour grain",
                    info="auto: on saturated colours of colour images only; off: luma only; on: even on a tinted frame",
                )
            with gr.Row():
                match_texture = gr.Checkbox(
                    value=True, label="Same visible grain on textured images",
                    info="textured or already grainy images hide part of the grain, so they get more (up to 1.4x); clean renders get exactly the slider's amount; never less",
                )
                film_type = gr.Radio(
                    choices=list(core.FILM_TYPES), value=core.FILM_DEFAULT, label="Film type",
                    info="print: a print or slide, most grain in the darks; negative: a negative scan, more in the highlights; grain is coarser in the shadows either way",
                )
            with gr.Row():
                softness = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Softness", info="slightly lower acutance than a digital render; 0 = off")
                halation = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Halation", info="red-orange glow around light sources; neutral on monochrome; 0 = off")
                bloom = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Bloom", info="soft glow around highlights in their own colour; 0 = off")
                highlight_rolloff = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Highlight roll-off", info="a film shoulder instead of a hard clip; white lands near 237/255 at 25, 226 at 50, 214 at 100; 0 = off")

        self.infotext_fields = [
            PasteField(enabled, _paste_enabled),
            PasteField(iso, _paste("iso", _paste_iso)),
            PasteField(grain, _paste("grain", float)),
            PasteField(colour_grain, _paste("chroma", str)),
            PasteField(match_texture, _paste("match", _paste_bool)),
            PasteField(film_type, _paste("film", str)),
            PasteField(softness, _paste_optic("soft")),
            PasteField(halation, _paste_optic("halation")),
            PasteField(bloom, _paste_optic("bloom")),
            PasteField(highlight_rolloff, _paste_optic("rolloff")),
        ]
        # Positional order == the hooks' parameter order.
        return [enabled, iso, grain, colour_grain, match_texture, film_type, softness, halation, bloom, highlight_rolloff]

    def process(self, p, enabled, iso, grain, colour_grain, match_texture, film_type, softness, halation, bloom, highlight_rolloff, **kwargs):
        if not enabled:
            return
        p.extra_generation_params[INFOTEXT_KEY] = infotext(iso, grain, colour_grain, match_texture, film_type, softness, halation, bloom, highlight_rolloff)

    @torch.inference_mode()
    def postprocess_image_after_composite(self, p, pp, enabled, iso, grain, colour_grain, match_texture, film_type, softness, halation, bloom, highlight_rolloff, **kwargs):
        if not enabled:
            return
        image = getattr(pp, "image", None)
        if not isinstance(image, Image.Image):
            return
        core = load_film_core(_EXTENSION_ROOT)
        iso = str(iso) if str(iso) in core.ISO_NAMES else core.ISO_DEFAULT
        colour_grain = str(colour_grain) if str(colour_grain) in core.CHROMA_MODES else "auto"
        film_type = str(film_type) if str(film_type) in core.FILM_TYPES else core.FILM_DEFAULT
        index = int(getattr(pp, "index", 0) or 0)
        seed = seed_for(p, index)

        x, mode = _to_tensor(image)
        work = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        t0 = time.perf_counter()
        out, stats = core.emulate(
            x, iso=iso, grain=float(grain), seed=seed, chroma=colour_grain, match_texture=bool(match_texture), film=film_type,
            softness=float(softness), halation=float(halation), bloom=float(bloom), rolloff=float(highlight_rolloff), work_device=work,
        )
        line = core.status_line(stats, seconds=time.perf_counter() - t0)

        logger.info(_ascii(f"Film emulation: {image.width}x{image.height} [{index + 1}] | {line}"))
        p.extra_generation_params[INFOTEXT_RESULT_KEY] = _ascii(line)
        pp.image = _to_image(out, mode)


_UI_PARAMS = ["enabled", "iso", "grain", "colour_grain", "match_texture", "film_type", "softness", "halation", "bloom", "highlight_rolloff"]
for _hook in (FilmEmulationScript.process, FilmEmulationScript.postprocess_image_after_composite):
    _params = [n for n in inspect.signature(_hook).parameters if n not in ("self", "p", "pp", "kwargs")]
    assert _params == _UI_PARAMS, (_hook.__name__, _params)
