"""ComfyUI-DeGrid — removes the 2px VAE pixel grid (Qwen Image / Wan 2.1 VAEs)."""

import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io, ui

from .degrid_core import degrid, NEGLIGIBLE_AMP


def _status_line(mode: str, stats: list) -> str:
    s = stats[0]
    amp = s["amp_255"]
    lim = s["limit"]
    src = "auto" if mode == "auto" else "manual"
    if amp < NEGLIGIBLE_AMP * 255.0:
        verdict = f"grid ≈ {amp:.2f}/255 — negligible, image already clean"
    else:
        verdict = f"grid ≈ {amp:.2f}/255 — removed (limit {lim:.3f} {src})"
    line = f"{verdict} · edges protected: {s['clipped_pct']:.1f}%"
    if len(stats) > 1:
        line += f" · batch of {len(stats)} (first shown)"
    return line


class VAEDeGrid(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDeGrid",
            display_name="VAE DeGrid (Nyquist Notch)",
            category="image/postprocessing",
            description=(
                "Removes the 2px pixel grid left by the Qwen Image / Wan 2.1 VAEs "
                "(Krea2, Qwen Image, Anima...). Wire directly after VAE Decode, "
                "before any sharpening or upscaling.\n\n"
                "Defaults are the zero-config path: leave mode on 'auto' and the node "
                "measures each image and calibrates itself. After a run, the node shows "
                "the measured grid strength, so you can see it did something even if "
                "the change is invisible at normal zoom.\n\n"
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
    def execute(cls, image, enabled, mode, limit, grid_gain, grid_view):
        if not enabled:
            return io.NodeOutput(
                image, torch.full_like(image, 0.5),
                ui=ui.PreviewText("bypassed (enabled = off)"),
            )
        cleaned, vis, stats = degrid(
            image, mode=mode, limit=limit, grid_gain=grid_gain, grid_view=grid_view,
        )
        return io.NodeOutput(cleaned, vis, ui=ui.PreviewText(_status_line(mode, stats)))


class DeGridExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VAEDeGrid]


async def comfy_entrypoint() -> DeGridExtension:
    return DeGridExtension()
