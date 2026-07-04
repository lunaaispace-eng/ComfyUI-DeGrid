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

## Usage

Wire it directly after **VAE Decode**, before any sharpening, deconvolution, or
upscaling (sharpening first would amplify the grid before removal).

| Widget | Default | Meaning |
|---|---|---|
| `enabled` | on | off = clean pass-through, for quick A/B |
| `mode` | `auto` | auto: measures the grid amplitude per image and sets the correction limit itself; `manual`: uses the `limit` widget |
| `limit` | 0.02 | manual mode only — max correction amplitude (0–1 scale); the VAE grid is typically 0.005–0.02 |
| `grid_gain` | 10 | amplification of the `removed_grid` debug output |

Outputs: the cleaned **image**, plus **removed_grid** — the subtracted
component, amplified and centered on gray. Preview it to verify: a uniform fine
grid means it is working; visible faces or fabric detail means the limit is too
high (manual mode, lower `limit`).

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
