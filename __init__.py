"""ComfyUI-DeGrid — removes the 2px VAE pixel grid (Qwen Image / Wan 2.1 VAEs)."""

import time

import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io, ui

from .degrid_core import NEGLIGIBLE_AMP, degrid, extract_grid, lattice_amp, status_line
from . import vae_enhance_core as enhance_core
from . import film_emulation_core as film_core

DEFAULT_THRESHOLD_255 = round(NEGLIGIBLE_AMP * 255.0, 2)  # 0.5


def _status_line(mode: str, stats: list, threshold_255: float = DEFAULT_THRESHOLD_255) -> str:
    return status_line(mode, stats, threshold=threshold_255 / 255.0)


def _console(line: str) -> None:
    """One line per run in the backend log (SwarmUI users only see the node text there)."""
    try:
        print(f"[DeGrid] {line}")
    except UnicodeEncodeError:
        print("[DeGrid] " + line.encode("ascii", "replace").decode("ascii"))


class VAEDeGrid(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDeGrid",
            display_name="VAE DeGrid (Nyquist Notch)",
            category="image/postprocessing",
            description=(
                "Removes the 2px pixel grid left by the Qwen Image / Qwen Image 2.1 "
                "/ Wan 2.1 VAEs (Krea2, Qwen Image, Anima...). Wire directly after "
                "VAE Decode, before any resize, sharpening or upscaling — and before "
                "a restorer like SeedVR2, which will otherwise treat the lattice as "
                "detail worth reconstructing.\n\n"
                "Defaults are the zero-config path: leave mode on 'auto' and the node "
                "measures each image and calibrates itself. It measures the lattice "
                "itself, not just how detailed the image is, so an image that never "
                "had a grid is reported as clean and passed through untouched. After "
                "a run, the node shows the measured grid strength and which "
                "orientation dominates.\n\n"
                "The removed_grid output shows WHAT was subtracted. The artifact is "
                "only 2px, so in 'full frame' view it looks like faint gray noise — "
                "that is correct behavior, not a failure. Switch grid_view to 4x/8x "
                "zoom to see the actual lattice pattern."
            ),
            search_aliases=["degrid", "notch", "grid artifact", "qwen vae", "krea2", "pixel grid"],
            inputs=[
                io.Image.Input("image", tooltip="Wire straight from VAE Decode."),
                io.Boolean.Input(
                    "enabled", default=True,
                    tooltip="Off = the image passes through completely untouched. "
                            "Use it to A/B compare with and without degrid.",
                ),
                io.Combo.Input(
                    "mode", options=["auto", "manual"], default="auto",
                    tooltip="auto (recommended): measures the grid strength of each "
                            "image and sets the removal limit itself — nothing to tune. "
                            "manual: uses the 'limit' value below instead; use it only "
                            "if auto visibly under- or over-corrects.",
                ),
                io.Float.Input(
                    "limit", default=0.02, min=0.0, max=0.10, step=0.001,
                    tooltip="MANUAL MODE ONLY (ignored in auto). Maximum per-pixel "
                            "correction on the 0-1 scale. The VAE grid is usually "
                            "0.005-0.02, so 0.02 is a good start. Too low = grid "
                            "partially survives in contrasty areas. Too high = fine "
                            "2-3px texture (pores, fabric) gets slightly softened.",
                ),
                io.Boolean.Input(
                    "skip_when_clean", default=True,
                    tooltip="Leave the image completely untouched when no grid is "
                            "actually there. The node measures the lattice directly "
                            "(its phase is locked to the VAE's output stride, so it "
                            "survives averaging while real detail cancels), and an "
                            "image with none — anything that has been through an "
                            "upscaler or a resize — is passed through bit-for-bit. "
                            "Turn this off only to force the filter to run regardless.",
                ),
                io.Float.Input(
                    "grid_gain", default=10.0, min=1.0, max=50.0, step=1.0,
                    tooltip="Brightness amplification of the removed_grid preview ONLY "
                            "— it never affects the cleaned image. Raise it if the "
                            "preview looks like flat gray and you want to see the "
                            "removed pattern more clearly.",
                ),
                io.Combo.Input(
                    "grid_view", options=["full frame", "4x zoom", "8x zoom"],
                    default="4x zoom",
                    tooltip="Framing of the removed_grid preview. The artifact is only "
                            "2px, so 'full frame' aliases into gray noise at node-preview "
                            "size — normal, but hard to read. '4x zoom' / '8x zoom' show "
                            "a magnified center crop where the actual 2px lattice is "
                            "visible. Preview only; the cleaned image is never cropped.",
                ),
                io.Float.Input(
                    "threshold", default=DEFAULT_THRESHOLD_255, min=0.05, max=5.0, step=0.05,
                    optional=True,
                    tooltip="Lattice amplitude, in /255 units, below which an image counts "
                            "as clean (see the status line's 'grid X/255'). Native Qwen-VAE "
                            "decodes measure about 0.5-2.5, images that went through an "
                            "upscaler or a resize about 0.1-0.2. Lower it (0.3) if a model "
                            "you know is gridded reports 'none detected'.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "cleaned", display_name="image",
                    tooltip="The degridded image — same size as the input. "
                            "Send this onward to sharpening/upscaling/save.",
                ),
                io.Image.Output(
                    "removed_grid", display_name="removed_grid",
                    tooltip="Visualization of what was subtracted (amplified by "
                            "grid_gain, centered on gray). Healthy result: a uniform "
                            "fine grid/noise texture. If you can recognize faces or "
                            "fabric here, the limit is too high. Not meant for further "
                            "processing — preview only.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, enabled, mode, limit, grid_gain, grid_view,
                skip_when_clean=True, threshold=DEFAULT_THRESHOLD_255):
        if not enabled:
            return io.NodeOutput(
                image, torch.full_like(image, 0.5),
                ui=ui.PreviewText("bypassed (enabled = off)"),
            )
        threshold = float(threshold) if threshold is not None else DEFAULT_THRESHOLD_255
        cleaned, vis, stats = degrid(
            image, mode=mode, limit=limit, grid_gain=grid_gain, grid_view=grid_view,
            skip_when_clean=skip_when_clean, threshold=threshold / 255.0,
        )
        line = _status_line(mode, stats, threshold)
        _console(line)
        return io.NodeOutput(cleaned, vis, ui=ui.PreviewText(line))


class VAEEnhance(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEEnhance",
            display_name="VAE Enhance (Flux.2 round trip)",
            category="image/postprocessing",
            description=(
                "Re-draws fine texture (skin pores, fur, hair) on a finished image by "
                "pushing it through a SECOND VAE: encode with the Flux.2 VAE, encode a "
                "slightly blurred copy too, move the latent away from the blurred one, "
                "decode. The decoder renders the added energy as texture, where a pixel "
                "unsharp mask would only add edge halo and grain.\n\n"
                "Wire the image from VAE Decode (through VAE DeGrid, or leave "
                "degrid_first on) and the vae from a Load VAE node holding "
                "flux2-vae.safetensors. Use it once, on the final image, before saving "
                "— not before an upscaler.\n\n"
                "gain 0.5 is the safe default: skin and fur improve from 0.5, regular "
                "fabric weave (denim, upholstery) moires from about 0.75. The mask keeps "
                "flat regions untouched (floor) and fades the gain out where a region "
                "already has texture (target). Keep input colour restores the exact hue "
                "and shading of the input, which the decoder otherwise shifts.\n\n"
                "Measured on the Flux.2 VAE only; any VAE with encode/decode is accepted "
                "and reported as untested."
            ),
            search_aliases=["enhance", "flux2 vae", "round trip", "skin texture", "detail", "plastic skin"],
            inputs=[
                io.Image.Input("image", tooltip="The finished image (from VAE Decode / VAE DeGrid)."),
                io.Vae.Input("vae", tooltip="The Flux.2 VAE (Load VAE -> flux2-vae.safetensors)."),
                io.Boolean.Input(
                    "enabled", default=True,
                    tooltip="Off = the image passes through completely untouched.",
                ),
                io.Float.Input(
                    "gain", default=enhance_core.GAIN, min=0.0, max=2.0, step=0.05,
                    tooltip="How far past the input to push the detail. 0 = a plain round trip. "
                            "0.5 is right for skin and fur; above 0.75 regular fabric weave "
                            "starts to moire and skin turns leathery from about 1.",
                ),
                io.Float.Input(
                    "sigma", default=enhance_core.SIGMA, min=0.5, max=2.0, step=0.1,
                    tooltip="Blur radius (px) that defines 'fine detail'. 1 targets pores and "
                            "fur; 2 pushes larger structure and brings block artifacts sooner.",
                ),
                io.Float.Input(
                    "mask_floor", default=enhance_core.FLOOR, min=0.0, max=10.0, step=0.5,
                    tooltip="Local texture (16px-block std of the 9px high-pass, /255) below "
                            "which a region gets nothing. Block medians on notched decodes: "
                            "clean sky 0.35, sand 1.4, flat painted wall 1.5, plastic skin "
                            "1.2-2, defocused fur 2, textured skin 3-6. 0.5 only drops clean "
                            "sky and keeps every kind of skin; 2.5-3.5 protects walls, sand and "
                            "defocus but loses plastic skin (use skin_only for that instead). "
                            "0 = off.",
                ),
                io.Float.Input(
                    "texture_target", default=enhance_core.TARGET, min=0.0, max=20.0, step=0.5,
                    tooltip="Gain fades out as a region approaches this texture level, so "
                            "skin that already has pores is not pushed into leather. 0 = off. "
                            "Turn it off for animals: in-focus fur sits above 9 and would be "
                            "protected away.",
                ),
                io.Boolean.Input(
                    "skin_only", default=True,
                    tooltip="Apply the gain only where the colour is skin-like (YCbCr skin "
                            "bands: fair, freckled and dark skin all pass; jeans, concrete and "
                            "painted walls do not; brown hair and beige fur do). This is what "
                            "keeps a plastic-skin portrait's walls and floor untouched, because "
                            "by texture alone plastic skin and a flat wall are the same. Turn it "
                            "off for animals and greyscale images.",
                ),
                io.Boolean.Input(
                    "tone_fix", default=True,
                    tooltip="Keep the input's colour and shading (17px low-pass) and take only "
                            "the fine detail from the round trip. The Flux.2 decoder otherwise "
                            "shifts skin by several levels of red. Costs nothing; leave it on.",
                ),
                io.Boolean.Input(
                    "degrid_first", default=True,
                    tooltip="Run the VAE DeGrid notch (auto mode) on the input before "
                            "enhancing. Required for gridded decodes: the blur removes the 2px "
                            "grid, so the detail direction would carry it and the extrapolation "
                            "would draw it back. An already clean image is passed through.",
                ),
                io.Boolean.Input(
                    "post_notch", default=False,
                    tooltip="Run the notch again on the output. The Flux.2 decoder has a "
                            "faint 2px checker of its own that grows with gain (about 0.2/255 "
                            "at gain 1). Costs 2px texture in the enhanced result.",
                ),
            ],
            outputs=[
                io.Image.Output("enhanced", display_name="image", tooltip="The enhanced image, same size as the input."),
                io.Image.Output("mask", display_name="mask", tooltip="Where the gain was applied (per latent pixel, white = full gain). Preview only."),
            ],
        )

    @classmethod
    def execute(cls, image, vae, enabled, gain, sigma, mask_floor, texture_target, skin_only, tone_fix, degrid_first, post_notch):
        if not enabled:
            return io.NodeOutput(image, torch.zeros_like(image), ui=ui.PreviewText("bypassed (enabled = off)"))
        t0 = time.perf_counter()
        x = image
        grid_in = removed = None
        if degrid_first:
            x, _, st = degrid(x, mode="auto")
            grid_in, removed = st[0]["amp_255"], not st[0]["skipped"]
        work = getattr(vae, "device", None)
        out, mask, stats = enhance_core.enhance(
            x, vae,
            sigma=float(sigma), gain=float(gain), floor=float(mask_floor), target=float(texture_target),
            skin_only=bool(skin_only), tone=bool(tone_fix), work_device=work if isinstance(work, torch.device) else None,
        )
        if post_notch:
            out, _, _ = degrid(out, mode="auto")
        grid_out = lattice_amp(extract_grid(out[..., :3].permute(0, 3, 1, 2).float()))[0][0].item() * 255.0
        line = enhance_core.status_line(
            stats, grid_in_255=grid_in, grid_in_removed=removed, grid_out_255=grid_out,
            seconds=time.perf_counter() - t0,
        )
        name, ok = enhance_core.describe_codec(vae)
        if not ok:
            line = f"untested VAE {name} · {line}"
        _console(line)
        return io.NodeOutput(out, mask, ui=ui.PreviewText(line))


class FilmEmulation(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FilmEmulation",
            display_name="Film Emulation",
            category="image/postprocessing",
            description=(
                "Makes the image look shot on film, as the LAST step before saving. Anything "
                "after it (a resize, an upscaler, a re-encode) resamples the grain into blobs, "
                "so wire it directly into Save.\n\n"
                "Grain: a seeded, spatially correlated noise field, coarser and stronger in the "
                "shadows, none on white photo borders. iso sets the grain's character (blob "
                "size, clumpiness, colour), never its amount; ISO 400 matches the grain Krea 2 "
                "draws itself when prompted for it. grain sets the amount alone, the same on "
                "every image and every ISO: 0 is off, 1 is under the eye's threshold, 50 reads "
                "as texture rather than noise (the picture looks more detailed, not noisier), "
                "100 is a bit noisy.\n\n"
                "Optics, each 0..100 and off at 0, run before the grain in linear light: "
                "softness (slightly lower acutance), halation (red glow around lights), bloom "
                "(soft glow around highlights), highlight_rolloff (a film shoulder instead of "
                "a hard clip). A white photo border is left out of every stage.\n\n"
                "Grey stays exactly grey: luma grain is identical on R, G and B. Colour grain "
                "only ever varies the colour a pixel already has, never adds a hue, and is "
                "switched off automatically on monochrome images. Judge the result at 100%."
            ),
            search_aliases=["film grain", "film emulation", "grain", "noise", "analog", "iso"],
            inputs=[
                io.Image.Input("image", tooltip="The finished image. This should be the last node before Save."),
                io.Boolean.Input("enabled", default=True, tooltip="Off = the image passes through completely untouched."),
                io.Combo.Input(
                    "iso", options=list(film_core.ISO_NAMES), default=film_core.ISO_DEFAULT,
                    tooltip="Grain character, not amount, named after real film stock. ISO 100 and 200: "
                            "fine, near-white, blend into skin as texture. ISO 400 (default): the "
                            "model's own grain, a cheap ISO 400 or a fine ISO 800 stock. ISO 800 and up: "
                            "larger, clumpier blobs that sit on the image, with more colour grain on "
                            "saturated colours.",
                ),
                io.Float.Input(
                    "grain", default=film_core.STRENGTH_DEFAULT, min=0.0, max=100.0, step=1.0,
                    tooltip="Grain amount at midtones: 0 = off, 1 = 0.3/255 (invisible), 50 = 3.3/255 "
                            "(at the midtone visibility threshold: texture, not noise), 100 = 8/255 "
                            "(grainy). film_type sets how much the other tones get.",
                ),
                io.Combo.Input(
                    "colour_grain", options=list(film_core.CHROMA_MODES), default="auto",
                    tooltip="auto: colour grain on saturated colours of colour images, none on monochrome "
                            "images (mean chroma under 8/255). off: luma grain only, always. on: colour "
                            "grain even on a faintly tinted frame (it still only varies the tint).",
                ),
                io.Boolean.Input(
                    "match_texture", default=True,
                    tooltip="Keep the visible amount of grain the same on every image. Texture an image "
                            "already has hides part of the grain added to it, so a textured or already "
                            "grainy image gets more (up to 1.4x) to show as much grain as a clean one. A "
                            "clean render gets exactly the grain setting's amount. Never reduces it. Off = "
                            "that amount on every image.",
                ),
                io.Combo.Input(
                    "film_type", options=list(film_core.FILM_TYPES), default=film_core.FILM_DEFAULT,
                    tooltip="Where the grain shows most. print (default): a print or a reversal slide, "
                            "most grain in the darks, less in the mids, little in the brights, none on "
                            "pure black or white. negative: a negative scan, more grain in the "
                            "highlights. Grain size follows exposure either way: coarser in the shadows.",
                ),
                io.Float.Input(
                    "softness", default=0.0, min=0.0, max=100.0, step=1.0,
                    tooltip="Slightly lower acutance than a digital render: a blend toward a 1 px Gaussian in linear light (lens plus emulsion). 0 = off.",
                ),
                io.Float.Input(
                    "halation", default=0.0, min=0.0, max=100.0, step=1.0,
                    tooltip="A red-orange glow around bright light sources: light reflected off the film base back into the red layer (CineStill has no anti-halation layer). Neutral on monochrome images. 0 = off.",
                ),
                io.Float.Input(
                    "bloom", default=0.0, min=0.0, max=100.0, step=1.0,
                    tooltip="A wide, soft glow around highlights in their own colour, as from a diffusion filter or lens glare. A bright area glows into its surroundings, not onto itself. 0 = off.",
                ),
                io.Float.Input(
                    "highlight_rolloff", default=0.0, min=0.0, max=100.0, step=1.0,
                    tooltip="A soft film shoulder on the brightest channel instead of a hard digital clip; hue is kept. Pure white lands at about 237/255 at 25, 226 at 50 and 214 at 100. Also takes the light halation and bloom add above white. 0 = off.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip="Grain pattern seed. Each image in a batch uses seed + its index.",
                ),
            ],
            outputs=[io.Image.Output("film", display_name="image", tooltip="The film-emulated image, same size as the input.")],
        )

    @classmethod
    def execute(cls, image, enabled, iso, grain, colour_grain, match_texture, film_type, softness, halation, bloom, highlight_rolloff, seed):
        if not enabled:
            return io.NodeOutput(image, ui=ui.PreviewText("bypassed (enabled = off)"))
        t0 = time.perf_counter()
        out, stats = film_core.emulate(
            image, iso=iso, grain=float(grain), seed=int(seed), chroma=colour_grain, match_texture=bool(match_texture), film=film_type,
            softness=float(softness), halation=float(halation), bloom=float(bloom), rolloff=float(highlight_rolloff),
        )
        line = film_core.status_line(stats, seconds=time.perf_counter() - t0)
        _console(line)
        return io.NodeOutput(out, ui=ui.PreviewText(line))


class DeGridExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VAEDeGrid, VAEEnhance, FilmEmulation]


async def comfy_entrypoint() -> DeGridExtension:
    return DeGridExtension()
