"""Pure-torch core for VAE DeGrid — no ComfyUI imports so it can be tested standalone.

Removes the 2px pixel grid left by the Qwen Image / Wan 2.1 VAEs using a
separable Nyquist notch, with an amplitude-limited correction so real edges
and fine texture pass through.

Filter: 9-tap alternating-sign binomial kernel -> 1D response sin^8(w/2).
2D combination (center - Bx - By + Bxy) factors into
(1 - sin^8(wx/2)) * (1 - sin^8(wy/2)):
  - exact zero response at any 2px-period pattern (stripes or checkerboard)
  - exact unity at DC with an 8th-order flat zero (no banding on gradients)
"""

import torch
import torch.nn.functional as F

_KERNEL = [1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0]
_NORM = 256.0
_PAD = 4

# Typical raw Qwen-VAE grid sits at 1-5/255; below this we call the image clean.
NEGLIGIBLE_AMP = 0.5 / 255.0


def extract_grid(x: torch.Tensor) -> torch.Tensor:
    """Extract the 2px-grid component of x.

    x: [B, C, H, W] float tensor. Returns Bx + By - Bxy, same shape —
    subtracting this from x is the full (unclamped) notch filter.
    """
    b, c, h, w = x.shape
    if h <= 2 * _PAD or w <= 2 * _PAD:
        return torch.zeros_like(x)
    k = torch.tensor(_KERNEL, dtype=x.dtype, device=x.device) / _NORM
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    bx = F.conv2d(F.pad(x, (_PAD, _PAD, 0, 0), mode="reflect"), kx, groups=c)
    by = F.conv2d(F.pad(x, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    # Bxy is separable: apply the vertical filter to Bx instead of a 9x9 conv
    bxy = F.conv2d(F.pad(bx, (0, 0, _PAD, _PAD), mode="reflect"), ky, groups=c)
    return bx + by - bxy


def _subsample(flat: torch.Tensor, max_samples: int = 1_000_000) -> torch.Tensor:
    n = flat.shape[-1]
    if n > max_samples:
        return flat[..., :: n // max_samples + 1]
    return flat


def auto_limit(
    corr: torch.Tensor,
    floor: float = 0.004,
    ceil: float = 0.05,
    mult: float = 3.0,
) -> torch.Tensor:
    """Per-image clamp limit from a robust estimate of the grid amplitude.

    Smooth regions dominate a photo, so the 75th percentile of |corr|
    approximates the artifact amplitude; edges are the outliers above it.
    Returns [B] tensor of limits.
    """
    flat = _subsample(corr.abs().reshape(corr.shape[0], -1))
    q = torch.quantile(flat.float(), 0.75, dim=1)
    return (q * mult).clamp(floor, ceil).to(corr.dtype)


def zoom_center(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Nearest-neighbor magnification of the center crop, for previewing
    the 2px lattice at a scale where it is actually visible.

    x: [B, H, W, C]. Returns roughly the same size (H//f*f, W//f*f).
    """
    b, h, w, c = x.shape
    ch, cw = max(2, h // factor), max(2, w // factor)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    crop = x[:, y0: y0 + ch, x0: x0 + cw, :]
    return crop.repeat_interleave(factor, dim=1).repeat_interleave(factor, dim=2)


def degrid(
    image: torch.Tensor,
    mode: str = "auto",
    limit: float = 0.02,
    grid_gain: float = 10.0,
    grid_view: str = "full frame",
):
    """Run the notch filter on a ComfyUI image batch.

    image: [B, H, W, C] in 0..1.
    grid_view: "full frame", "4x zoom" or "8x zoom" — framing of the
    removed-grid visualization (zoom = magnified center crop so the 2px
    lattice is visible in a node preview).

    Returns (cleaned, grid_vis, stats): cleaned matches the input shape and
    dtype; grid_vis is the removed component amplified and centered on 0.5
    gray; stats is a list of per-image dicts with:
      amp_255     robust grid amplitude estimate, in /255 units
      limit       clamp limit actually applied (auto or manual)
      clipped_pct percent of pixels where the correction hit the clamp
                  (those are real edges being protected)
    """
    orig_dtype = image.dtype
    x = image.permute(0, 3, 1, 2).contiguous().float()
    corr = extract_grid(x)

    flat = _subsample(corr.abs().reshape(corr.shape[0], -1)).float()
    amp = torch.quantile(flat, 0.75, dim=1)  # [B]

    if mode == "auto":
        lim = auto_limit(corr)  # [B]
    else:
        lim = torch.full((x.shape[0],), float(limit), dtype=corr.dtype, device=corr.device)
    lim_b = lim.view(-1, 1, 1, 1)
    clipped = (corr.abs() > lim_b).float().mean(dim=(1, 2, 3)) * 100.0  # [B] %
    corr = corr.clamp(-lim_b, lim_b)

    cleaned = (x - corr).clamp(0.0, 1.0)
    vis = (corr * float(grid_gain) + 0.5).clamp(0.0, 1.0)
    cleaned = cleaned.permute(0, 2, 3, 1).contiguous().to(orig_dtype)
    vis = vis.permute(0, 2, 3, 1).contiguous().to(orig_dtype)

    if grid_view == "4x zoom":
        vis = zoom_center(vis, 4)
    elif grid_view == "8x zoom":
        vis = zoom_center(vis, 8)

    stats = [
        {
            "amp_255": amp[i].item() * 255.0,
            "limit": lim[i].item(),
            "clipped_pct": clipped[i].item(),
        }
        for i in range(x.shape[0])
    ]
    return cleaned, vis, stats
