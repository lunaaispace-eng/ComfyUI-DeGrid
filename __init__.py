"""ComfyUI-DeGrid — removes the 2px VAE pixel grid (Qwen Image / Wan 2.1 VAEs)."""

import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

from .degrid_core import degrid


class VAEDeGrid(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDeGrid",
            display_name="VAE DeGrid (Nyquist Notch)",
            category="image/postprocessing",
            description=(
                "Removes the 2px pixel grid left by the Qwen Image / Wan 2.1 VAEs. "
                "Wire directly after VAE Decode, before any sharpening or upscaling. "
                "Auto mode calibrates the correction limit per image; the removed_grid "
                "output shows what was subtracted (amplified) for verification."
            ),
            search_aliases=["degrid", "notch", "grid artifact", "qwen vae", "krea2", "pixel grid"],
            inputs=[
                io.Image.Input("image"),
                io.Boolean.Input("enabled", default=True,
                                 tooltip="Off = pass the image through unchanged (quick A/B)"),
                io.Combo.Input("mode", options=["auto", "manual"], default="auto",
                               tooltip="auto: per-image calibration of the correction limit; "
                                       "manual: use the limit widget below"),
                io.Float.Input("limit", default=0.02, min=0.0, max=0.10, step=0.001,
                               tooltip="Manual mode only. Max correction amplitude on the 0-1 "
                                       "scale; the VAE grid is typically 0.005-0.02. Higher = "
                                       "stronger removal but softens fine detail near 2-3px."),
                io.Float.Input("grid_gain", default=10.0, min=1.0, max=50.0, step=1.0,
                               tooltip="Amplification of the removed_grid debug output. A uniform "
                                       "fine grid there = working correctly; visible faces/fabric "
                                       "detail = limit too high."),
            ],
            outputs=[
                io.Image.Output("cleaned", display_name="image"),
                io.Image.Output("removed_grid", display_name="removed_grid"),
            ],
        )

    @classmethod
    def execute(cls, image, enabled, mode, limit, grid_gain):
        if not enabled:
            return io.NodeOutput(image, torch.full_like(image, 0.5))
        cleaned, vis = degrid(image, mode=mode, limit=limit, grid_gain=grid_gain)
        return io.NodeOutput(cleaned, vis)


class DeGridExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VAEDeGrid]


async def comfy_entrypoint() -> DeGridExtension:
    return DeGridExtension()
