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


def auto_limit(
    corr: torch.Tensor,
    floor: float = 0.004,
    ceil: float = 0.05,
    mult: float = 3.0,
    max_samples: int = 1_000_000,
) -> torch.Tensor:
    """Per-image clamp limit from a robust estimate of the grid amplitude.

    Smooth regions dominate a photo, so the 75th percentile of |corr|
    approximates the artifact amplitude; edges are the outliers above it.
    Returns [B] tensor of limits.
    """
    flat = corr.abs().reshape(corr.shape[0], -1)
    n = flat.shape[1]
    if n > max_samples:
        flat = flat[:, :: n // max_samples + 1]
    q = torch.quantile(flat.float(), 0.75, dim=1)
    return (q * mult).clamp(floor, ceil).to(corr.dtype)


def degrid(
    image: torch.Tensor,
    mode: str = "auto",
    limit: float = 0.02,
    grid_gain: float = 10.0,
):
    """Run the notch filter on a ComfyUI image batch.

    image: [B, H, W, C] in 0..1. Returns (cleaned, grid_vis), same shape/dtype.
    grid_vis is the removed component amplified and centered on 0.5 gray.
    """
    orig_dtype = image.dtype
    x = image.permute(0, 3, 1, 2).contiguous().float()
    corr = extract_grid(x)
    if mode == "auto":
        lim = auto_limit(corr).view(-1, 1, 1, 1)
        corr = corr.clamp(-lim, lim)
    else:
        corr = corr.clamp(-float(limit), float(limit))
    cleaned = (x - corr).clamp(0.0, 1.0)
    vis = (corr * float(grid_gain) + 0.5).clamp(0.0, 1.0)
    cleaned = cleaned.permute(0, 2, 3, 1).contiguous().to(orig_dtype)
    vis = vis.permute(0, 2, 3, 1).contiguous().to(orig_dtype)
    return cleaned, vis
