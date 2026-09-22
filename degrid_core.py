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


def lattice_amp(corr: torch.Tensor):
    """Phase-locked amplitude of the 2px lattice in corr.

    The VAE grid is tied to the decoder's output stride, so its phase is
    constant across the whole frame. Averaging each of the four (y%2, x%2)
    sublattices therefore keeps the grid and averages incoherent detail away:
    real texture that merely lands in the notch band cancels over millions of
    pixels, the lattice does not.

    This is the honest "is there a grid here" measurement. The 75th percentile
    of |corr| is NOT — it tracks how much fine detail an image has, not how
    much lattice, and on a busy but clean image it reads higher than on a
    gridded flat one (measured 2026-09-20, see STATUS.md).

    Returns (p2p, checker, vstripe, hstripe), each [B] in 0..1 units, taken
    from whichever colour channel carries the most.
    """
    h = corr.shape[2] // 2 * 2
    w = corr.shape[3] // 2 * 2
    if h < 2 or w < 2:
        z = torch.zeros(corr.shape[0], dtype=corr.dtype, device=corr.device)
        return z, z, z, z
    c = corr[:, :, :h, :w]
    # [B, C, 4] = means of the m00, m01, m10, m11 sublattices
    m = torch.stack(
        [c[:, :, i::2, j::2].mean(dim=(2, 3)) for i in (0, 1) for j in (0, 1)],
        dim=-1,
    )
    p2p = (m.amax(-1) - m.amin(-1)).amax(-1)
    m00, m01, m10, m11 = (m[..., k] for k in range(4))
    checker = (((m00 + m11) - (m01 + m10)) / 2).abs().amax(-1)
    vstripe = (((m00 + m10) - (m01 + m11)) / 2).abs().amax(-1)
    hstripe = (((m00 + m01) - (m10 + m11)) / 2).abs().amax(-1)
    return p2p, checker, vstripe, hstripe


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


# Auto mode raises the clamp until the lattice left behind is under this
# fraction of the clean threshold (0.25/255 at the default threshold).
AUTO_TARGET_FRACTION = 0.5
AUTO_LADDER_RATIO = 1.5


def residual_after_clamp(corr: torch.Tensor, lim: torch.Tensor) -> torch.Tensor:
    """Phase-locked lattice that survives clamping corr to +-lim. [B] in 0..1 units.

    Exact, not an estimate: the sublattice means are linear and the notch has
    unit response at the three 2px frequencies, so the lattice in (x - clamp(corr))
    equals the lattice in (corr - clamp(corr)). No second filter pass needed.
    """
    lim_b = lim.view(-1, 1, 1, 1)
    return lattice_amp(corr - corr.clamp(-lim_b, lim_b))[0]


def auto_limit_targeted(
    corr: torch.Tensor,
    threshold: float = NEGLIGIBLE_AMP,
    floor: float = 0.004,
    ceil: float = 0.05,
    mult: float = 3.0,
    ratio: float = AUTO_LADDER_RATIO,
    target_fraction: float = AUTO_TARGET_FRACTION,
) -> torch.Tensor:
    """Per-image clamp limit that actually clears the grid.

    Starts from auto_limit() (a robust guess from the notch band's 75th
    percentile) and, image by image, steps it up by `ratio` until the residual
    lattice drops under `threshold * target_fraction`, or `ceil` is reached.
    On a Krea 2 decode the first or second rung is enough; a Qwen Image 2.1
    decode carries a heavier notch-band tail and needs 0.03-0.05, where the
    percentile guess alone left a third of the grid behind (measured 2026-09-21).
    Each rung costs a clamp and four sublattice means, nothing more.
    Returns [B] tensor of limits.
    """
    lim = auto_limit(corr, floor, ceil, mult)
    target = float(threshold) * float(target_fraction)
    chosen = lim.clone()
    done = torch.zeros(corr.shape[0], dtype=torch.bool, device=corr.device)
    cand = lim.clone()
    while True:
        resid = residual_after_clamp(corr, cand)
        chosen = torch.where(done, chosen, cand)
        done = done | (resid <= target)
        if bool((done | (cand >= ceil)).all()):  # every entry is satisfied or has nowhere left to go
            break
        cand = torch.where(done, cand, (cand * ratio).clamp(max=ceil))
    return chosen


# Auto mode calibrates per tile of this many pixels, then blends the tile limits
# into a smooth per-pixel map. The grid rides on texture: on a Qwen Image 2.1
# desert it measured 0.7/255 in the sky and 8.8/255 on the ground, and one
# frame-wide limit high enough for the ground shaved the twigs against the sky.
AUTO_TILE = 256


def _pad_to_multiple(x: torch.Tensor, t: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % t, (-w) % t
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x


def _tile_quantile_abs(corr: torch.Tensor, t: int, q: float = 0.75) -> torch.Tensor:
    """q-quantile of |corr| per t x t tile (all channels pooled). [B, th, tw]."""
    b, c, hp, wp = corr.shape
    th, tw = hp // t, wp // t
    v = corr.abs().reshape(b, c, th, t, tw, t).permute(0, 2, 4, 1, 3, 5).reshape(b, th, tw, c * t * t)
    k = max(1, int(round(q * v.shape[-1])))
    return v.kthvalue(k, dim=-1).values  # no element-count limit, unlike torch.quantile


def _tile_lattice_p2p(d: torch.Tensor, t: int) -> torch.Tensor:
    """Phase-locked 2px lattice per tile: peak-to-peak of the four sublattice
    means inside each t x t tile, max over channels. [B, th, tw], 0..1 units."""
    b, c, hp, wp = d.shape
    th, tw = hp // t, wp // t
    m = d.reshape(b, c, th, t // 2, 2, tw, t // 2, 2).mean(dim=(3, 6))  # [B, C, th, 2, tw, 2]
    m = m.permute(0, 1, 2, 4, 3, 5).reshape(b, c, th, tw, 4)
    return (m.amax(-1) - m.amin(-1)).amax(1)


def auto_limit_map(
    corr: torch.Tensor,
    threshold: float = NEGLIGIBLE_AMP,
    tile: int = AUTO_TILE,
    floor: float = 0.004,
    ceil: float = 0.05,
    mult: float = 3.0,
    ratio: float = AUTO_LADDER_RATIO,
    target_fraction: float = AUTO_TARGET_FRACTION,
):
    """Per-pixel clamp limit: auto_limit_targeted() run per tile, then blended.

    Each tile gets its own percentile guess and its own ladder, so a flat sky
    settles near the floor while textured ground climbs to the ceiling, and the
    tile limits are bilinearly interpolated from tile centres to pixels so the
    clamp has no seams. Images smaller than one tile fall back to one global
    limit. Returns (limit_map [B, 1, H, W], tile_limits [B, th, tw]).
    """
    b, c, h, w = corr.shape
    t = max(2, int(tile) // 2 * 2)
    if t < 8 or h < t or w < t:
        lim = auto_limit_targeted(corr, threshold, floor, ceil, mult, ratio, target_fraction)
        return lim.view(b, 1, 1, 1).expand(b, 1, h, w), lim.view(b, 1, 1)
    padded = _pad_to_multiple(corr, t)
    target = float(threshold) * float(target_fraction)
    cand = (_tile_quantile_abs(padded, t) * mult).clamp(floor, ceil)  # [B, th, tw]
    chosen = cand.clone()
    done = torch.zeros_like(cand, dtype=torch.bool)
    while True:
        lim_px = cand.repeat_interleave(t, 1).repeat_interleave(t, 2).unsqueeze(1)  # block-constant [B, 1, Hp, Wp]
        resid = _tile_lattice_p2p(padded - padded.clamp(min=-lim_px, max=lim_px), t)
        chosen = torch.where(done, chosen, cand)
        done = done | (resid <= target)
        if bool((done | (cand >= ceil)).all()):  # every entry is satisfied or has nowhere left to go
            break
        cand = torch.where(done, cand, (cand * ratio).clamp(max=ceil))
    # Blend for seamless clamping, but never below a tile's own calibrated limit:
    # the ramp from a strong tile runs outward into its neighbours, so the residual
    # target still holds inside every tile.
    blocks = chosen.repeat_interleave(t, 1).repeat_interleave(t, 2).unsqueeze(1)
    smooth = F.interpolate(chosen.unsqueeze(1), size=padded.shape[-2:], mode="bilinear", align_corners=False)
    lim_map = torch.maximum(smooth, blocks)
    return lim_map[:, :, :h, :w], chosen


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
    skip_when_clean: bool = True,
    threshold: float = NEGLIGIBLE_AMP,
    tile: int = AUTO_TILE,
):
    """Run the notch filter on an image batch.

    image: [B, H, W, C] in 0..1. C > 3 (RGBA) is treated as colour plus
    passthrough channels: only the first three are measured and filtered.
    grid_view: "full frame", "4x zoom" or "8x zoom" — framing of the
    removed-grid visualization (zoom = magnified center crop so the 2px
    lattice is visible in a node preview).

    skip_when_clean: leave an image completely untouched when no phase-locked
    lattice is detected. Without it the notch still shaves ~1/255 of genuine
    fine texture off an image that never had a grid.

    threshold: lattice amplitude (0..1 units) below which an image counts as
    clean. Defaults to NEGLIGIBLE_AMP (0.5/255).

    In "auto" mode the clamp limit is calibrated per `tile` x `tile` pixels
    and raised there until the lattice left behind is under half of
    `threshold`, then blended into a smooth per-pixel map (see
    auto_limit_map); "manual" uses `limit` everywhere as given.

    Returns (cleaned, grid_vis, stats): cleaned matches the input shape and
    dtype; grid_vis is the removed component amplified and centered on 0.5
    gray; stats is a list of per-image dicts with:
      amp_255     phase-locked lattice amplitude (peak-to-peak), /255 units
      texture_255 75th percentile of |correction| — a detail measure, kept
                  for diagnostics only; it is NOT a grid measure
      checker_255 / vstripe_255 / hstripe_255  the lattice broken into its
                  checkerboard and two stripe components
      limit       clamp limit applied: the manual value, or in auto mode the
                  median tile limit
      limit_min / limit_max   range of the tile limits in auto mode (equal to
                  limit in manual mode)
      clipped_pct percent of pixels where the correction hit the clamp
                  (those are real edges being protected)
      residual_255 phase-locked lattice left in the cleaned image, /255 units;
                  above ~threshold means the clamp limit was too low to
                  remove the grid, not that the notch missed it
      skipped     True if the image was passed through untouched as clean
    """
    orig_dtype = image.dtype
    x_all = image.permute(0, 3, 1, 2).contiguous().float()
    # Colour only: an RGBA decode (Qwen Image 2.1) keeps its alpha plane untouched,
    # and alpha takes no part in the measurement or the calibration either.
    x, extra = (x_all[:, :3], x_all[:, 3:]) if x_all.shape[1] > 3 else (x_all, None)
    corr = extract_grid(x)

    amp, chk, vst, hst = lattice_amp(corr)  # each [B]
    flat = _subsample(corr.abs().reshape(corr.shape[0], -1)).float()
    texture = torch.quantile(flat, 0.75, dim=1)  # [B]

    if mode == "auto":
        lim_map, tile_lims = auto_limit_map(corr, threshold=threshold, tile=tile)  # [B, 1, H, W], [B, th, tw]
        lim = tile_lims.reshape(x.shape[0], -1).median(dim=1).values  # [B] representative: median tile limit
        lim_min = tile_lims.reshape(x.shape[0], -1).amin(dim=1)
        lim_max = tile_lims.reshape(x.shape[0], -1).amax(dim=1)
    else:
        lim = torch.full((x.shape[0],), float(limit), dtype=corr.dtype, device=corr.device)
        lim_map = lim.view(-1, 1, 1, 1)
        lim_min = lim_max = lim
    clipped = (corr.abs() > lim_map).float().mean(dim=(1, 2, 3)) * 100.0  # [B] %
    corr_raw = corr
    corr = corr.clamp(min=-lim_map, max=lim_map)
    # Lattice left in the cleaned image = lattice(x) - lattice(corr): the
    # sublattice means are linear, and extract_grid has unit response at the
    # three 2px frequencies, so measuring the unremoved part is exact.
    residual = lattice_amp(corr_raw - corr)[0]  # [B]

    # No lattice -> subtract nothing. The notch is cheap but not free: on a
    # clean, detailed image it still removes ~1/255 of real high-frequency
    # detail, and a SeedVR2 / upscaler output has no grid left to remove.
    skipped = amp < float(threshold)
    if skip_when_clean:
        corr = torch.where(skipped.view(-1, 1, 1, 1), torch.zeros_like(corr), corr)
        clipped = torch.where(skipped, torch.zeros_like(clipped), clipped)
        residual = torch.where(skipped, amp, residual)

    cleaned = (x - corr).clamp(0.0, 1.0)
    vis = (corr * float(grid_gain) + 0.5).clamp(0.0, 1.0)
    if extra is not None:
        cleaned = torch.cat([cleaned, extra], dim=1)
        vis = torch.cat([vis, torch.ones_like(extra)], dim=1)  # opaque preview
    cleaned = cleaned.permute(0, 2, 3, 1).contiguous().to(orig_dtype)
    vis = vis.permute(0, 2, 3, 1).contiguous().to(orig_dtype)

    if grid_view == "4x zoom":
        vis = zoom_center(vis, 4)
    elif grid_view == "8x zoom":
        vis = zoom_center(vis, 8)

    stats = [
        {
            "amp_255": amp[i].item() * 255.0,
            "texture_255": texture[i].item() * 255.0,
            "checker_255": chk[i].item() * 255.0,
            "vstripe_255": vst[i].item() * 255.0,
            "hstripe_255": hst[i].item() * 255.0,
            "limit": lim[i].item(),
            "limit_min": lim_min[i].item(),
            "limit_max": lim_max[i].item(),
            "clipped_pct": clipped[i].item(),
            "residual_255": residual[i].item() * 255.0,
            "skipped": bool(skipped[i].item()) and skip_when_clean,
        }
        for i in range(x.shape[0])
    ]
    return cleaned, vis, stats


def status_line(mode: str, stats: list, threshold: float = NEGLIGIBLE_AMP) -> str:
    """One-line human-readable verdict for a degrid() stats list.

    Shared by every front end (ComfyUI node text, Forge Neo console/infotext)
    so they all describe a result in the same words.
    """
    s = stats[0]
    amp = s["amp_255"]
    lim = s["limit"]
    src = "auto" if mode == "auto" else "manual"
    if amp < float(threshold) * 255.0:
        state = "passed through untouched" if s["skipped"] else "filtered anyway"
        verdict = f"grid {amp:.2f}/255 — none detected, {state}"
    else:
        # name the dominant orientation: it says which stage left the lattice
        parts = (
            ("checker", s["checker_255"]),
            ("V-stripe", s["vstripe_255"]),
            ("H-stripe", s["hstripe_255"]),
        )
        kind = max(parts, key=lambda p: p[1])[0]
        residual = s.get("residual_255", 0.0)
        lo, hi = s.get("limit_min", lim), s.get("limit_max", lim)
        lim_txt = f"{lim:.3f}" if abs(hi - lo) < 5e-4 else f"{lo:.3f}-{hi:.3f}"
        if residual >= float(threshold) * 255.0:
            verdict = f"grid {amp:.2f}/255 ({kind}) — partially removed, {residual:.2f}/255 left (limit {lim_txt} {src} too low)"
        else:
            verdict = f"grid {amp:.2f}/255 ({kind}) — removed (limit {lim_txt} {src})"
    # no colon in the line: Forge JSON-quotes any infotext value containing one
    line = f"{verdict} · edges protected {s['clipped_pct']:.1f}%"
    if len(stats) > 1:
        line += f" · batch of {len(stats)} (first shown)"
    return line
