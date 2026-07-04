# ComfyUI-DeGrid

One node: **VAE DeGrid (Nyquist Notch)** — removes the 2px pixel grid that the
Qwen Image VAE (and, to a lesser extent, the Wan 2.1 VAE) leaves across decoded
images. Affects Krea2, Qwen Image, Anima and anything else built on those VAEs.

The artifact is easy to miss at 100% zoom, but it gets amplified by any
sharpening or upscaling applied afterwards. This node erases it exactly, with
auto-calibration so there is nothing to tune.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/lunaaispace-eng/ComfyUI-DeGrid
```

No dependencies beyond torch (no OpenGL/GLFW — works headless). Restart ComfyUI
and search for **degrid**.

## Quick start

1. Wire **VAE Decode → VAE DeGrid → (everything else)**. It must sit *before*
   any sharpening, deconvolution, or upscaling — those amplify the grid, so
   remove it first.
2. Leave the defaults. `auto` mode measures each image and calibrates itself.
3. Run once. The node displays a status line, e.g.:

   ```
   grid ≈ 2.10/255 — removed (limit 0.019 auto) · edges protected: 1.2%
   ```

   That readout is your confirmation it worked — you don't need to pixel-peep.

## Settings

| Widget | Default | What it does |
|---|---|---|
| `enabled` | on | Off = the image passes through completely untouched. Flip it for a quick A/B comparison. |
| `mode` | `auto` | **auto (recommended):** measures the grid strength per image and sets the removal limit itself — nothing to tune, adapts to different VAEs and content. **manual:** uses the `limit` widget instead. Only switch if auto visibly under- or over-corrects. |
| `limit` | 0.02 | **Manual mode only** (ignored in auto). Maximum per-pixel correction on the 0–1 scale. The VAE grid is usually 0.005–0.02. Too low → grid partially survives in contrasty areas. Too high → fine 2–3px texture (skin pores, fabric weave) gets slightly softened. |
| `grid_gain` | 10 | Brightness amplification of the `removed_grid` preview **only** — never affects the cleaned image. Raise it if the preview looks like flat gray. |
| `grid_view` | `4x zoom` | Framing of the `removed_grid` preview. `full frame` shows the whole image (reads as gray noise at preview size — see below). `4x zoom` / `8x zoom` show a magnified center crop where the actual 2px lattice is visible. Preview only; the cleaned image is never cropped. |

### Status line

After each run the node shows what it measured:

- **`grid ≈ X/255`** — the estimated artifact amplitude. The raw Qwen-VAE grid
  is typically 1–5/255. Below ~0.5/255 the node reports the image as already
  clean (the filter then changes almost nothing — that's correct).
- **`limit N (auto|manual)`** — the correction cap that was applied.
- **`edges protected: N%`** — percentage of pixels where the correction hit the
  cap. Those are real edges/detail being passed through unsoftened. A few
  percent is normal; a very high number means lots of legitimate
  high-frequency content (or a manual limit set too low).

## Reading the removed_grid preview

**"It's just gray noise — is it even doing anything?"** Yes. That is exactly
what success looks like, and here is why: the artifact is a 2-pixel pattern.
A node preview shows a 1728px image at a few hundred pixels wide, so a 2px
lattice is far below what the thumbnail can render — it aliases into uniform
gray "noise". The information is real; the zoom level just can't show it.

Two ways to actually see it:

- Set `grid_view` to `4x zoom` or `8x zoom` (default is 4x): the preview
  becomes a magnified center crop and the regular lattice pattern is plainly
  visible.
- Or open the preview image at 100%+ zoom.

What to look for:

| removed_grid shows | Meaning |
|---|---|
| Uniform fine grid / speckle, brighter over textured areas | Working correctly |
| Nearly flat gray | Little or no grid in this image (check the status line — likely `negligible`) |
| Recognizable faces, fabric, edges | Limit too high — switch to manual and lower `limit` |

A faint silhouette of the subject is normal (the artifact is slightly stronger
over detailed areas). Recognizable *detail* is not.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Grid still visible in contrasty areas after filtering | `mode: manual`, raise `limit` toward 0.03–0.04 |
| Fine texture (pores, weave) looks softened | `mode: manual`, lower `limit` toward 0.01 |
| Status says `negligible` but you see a grid | The grid may be coming from a later node (sharpener, upscaler) — this node only fixes what the VAE decode produced. Check the chain order. |
| removed_grid looks like flat gray | Raise `grid_gain`, or the image simply has no grid |

## How it works

A separable Nyquist notch: 9-tap alternating-sign binomial kernel
(1D response sin⁸(ω/2)), combined as `center − Bx − By + Bxy`, which factors
into `(1 − sin⁸(ωx/2))(1 − sin⁸(ωy/2))`. That is an exact zero for any
2px-period pattern — vertical stripes, horizontal stripes, or checkerboard —
and exact unity at DC with an 8th-order flat zero, so gradients pass through
without banding.

The correction is then amplitude-clamped before subtraction, so strong real
edges and legitimate fine texture pass through unsoftened — only the
low-amplitude artifact band is removed. In `auto` mode the clamp limit is
estimated per image from a robust percentile of the extracted grid component,
so it adapts to different VAEs, LoRA stacks, and content automatically.

Based on the GLSL notch-filter approach shared by
[u/Haiku-575 on r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1umwhq7/2px_pixel_grid_on_krea2_from_vae_and_how_to/),
reimplemented in pure PyTorch with a narrower 9-tap kernel, amplitude limiting,
and per-image auto-calibration.

## License

Apache-2.0
