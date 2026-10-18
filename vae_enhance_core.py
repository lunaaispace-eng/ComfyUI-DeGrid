"""Pure-torch core for the VAE enhancement (detail extrapolation through a second VAE).

No ComfyUI / Forge imports, so the same file drives the ComfyUI node and the
Forge Neo extension and can be tested standalone with a fake codec.

The idea
--------
Encode the finished image with a *different* VAE (Flux.2 is the one this was
developed and measured on), encode a slightly blurred copy of it too, and
extrapolate away from the blurred latent before decoding::

    z      = E(x)
    z_blur = E(blur_sigma(x))
    out    = D(z + gain * mask * (z - z_blur))

``z - z_blur`` is what the encoder considers fine detail, expressed in the
channel space where sub-8px structure (pores, fur, hair) actually lives; a
spatial kernel on the 16px latent grid could not reach it. The decoder then
*renders* the added energy: at matched high-frequency energy a pixel unsharp
mask produces edge halo and uniform grain, this produces skin and fur texture.

Two guards make it usable and one repair makes it faithful:

* ``mask``: per latent pixel, from the local texture of the input. A *floor*
  keeps flat regions (walls, sky, defocused areas) untouched, because the
  direction there is encoder noise and the decoder's own lattice. A *target*
  fades the gain out where the region already carries texture, so skin that
  already has pores does not turn into leather. Both are in the units of the
  status line: std of the 9px-high-passed luma, /255.
* ``tone_fix``: the decoder shifts hue and shading along with the detail
  (Flux.2: red down 3-6 levels on skin at gain 0.75-1). Keeping the input's
  17px low-pass and taking only the high-pass from the result restores colour
  exactly and recovers ~2 dB.

Run the notch (degrid_core) on the input *first*: blurring removes the 2px
grid, so ``z - z_blur`` would otherwise carry the lattice and the extrapolation
re-draws it. The front ends do that; this module only asks for a clean image.

Known limits (measured, see the README): a fixed gain amplifies regular fine
texture (denim, woven fabric) into moire from gain ~0.75; the energy-based mask
cannot tell a periodic weave from irregular fur. Keep the gain at 0.5 unless the
image has no such fabric. The decoder does not smooth an over-textured input; a
round trip reproduces it and the extrapolation makes it stronger.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Protocol

import torch
import torch.nn.functional as F

SIGMA = 1.0
GAIN = 0.5
# Local texture, 16px-block std of the 9px high-pass luma, /255, block medians
# measured on notched Krea 2 decodes: clean sky 0.35, sand 1.4, flat painted wall
# 1.5, plastic skin 1.7-2.0, defocused fur 2.0, noisy sky 2.5, textured skin
# 3-6, fur/hair/fabric 7-16. Energy cannot separate plastic skin from a wall,
# sand or defocus, so the floor only removes clean sky by default (the ramp is
# 1 wide: 0.5 -> 1.5, so plastic-skin blocks at 1.2-1.7 keep 0.7-1.0) and the
# skin-tone term does the separating for portraits. Raise the floor to 2.5-3.5
# to protect walls when the skin is not plastic.
FLOOR = 0.5  # /255; below it the mask is 0 (0 = off)
TARGET = 9.0  # /255; at and above it the mask is 0 again (0 = off)
SKIN_SOFT = 8.0  # width of the soft edges of the YCbCr skin-tone bands
TONE_RADIUS = 17  # px, box low-pass kept from the input
HF_RADIUS = 9  # px, box high-pass that defines "texture" everywhere in this module
MASK_SMOOTH = 3  # latent px, box smoothing of the mask
FLOOR_RAMP = 1.0  # /255; the mask ramps 0..1 over [floor, floor + FLOOR_RAMP]

_LUMA = (0.299, 0.587, 0.114)


class Codec(Protocol):
    """What both front ends' VAE objects already provide (ComfyUI ``comfy.sd.VAE``
    and Forge Neo ``backend.patcher.vae.VAE``): image tensors ``[B, H, W, C]`` in
    0..1 in, latents out, and back."""

    def encode(self, pixels: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor: ...


# -- building blocks ------------------------------------------------------------


def luma(x_bchw: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_LUMA, dtype=x_bchw.dtype, device=x_bchw.device).view(1, 3, 1, 1)
    return (x_bchw[:, :3] * w).sum(1, keepdim=True)


def box(x_bchw: torch.Tensor, k: int) -> torch.Tensor:
    """k x k box blur with reflect padding (replicate when the image is too small)."""
    if k <= 1:
        return x_bchw
    p = k // 2
    mode = "reflect" if min(x_bchw.shape[-2:]) > p else "replicate"
    return F.avg_pool2d(F.pad(x_bchw, (p, p, p, p), mode=mode), k, stride=1)


def gaussian_blur(x_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x_bchw
    radius = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-radius, radius + 1, dtype=x_bchw.dtype, device=x_bchw.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x_bchw.shape[1]
    mode = "reflect" if min(x_bchw.shape[-2:]) > radius else "replicate"
    y = F.pad(x_bchw, (radius, radius, radius, radius), mode=mode)
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return y


def highpass_luma_255(x_bchw: torch.Tensor) -> torch.Tensor:
    """Luma minus its 9px box blur, in 8-bit units: the texture measure. [B, 1, H, W]"""
    L = luma(x_bchw) * 255.0
    return L - box(L, HF_RADIUS)


def hf_energy_255(x_bchw: torch.Tensor) -> torch.Tensor:
    """Frame-wide texture level: std of the high-pass, /255. [B]"""
    hp = highpass_luma_255(x_bchw)
    return hp.reshape(hp.shape[0], -1).std(dim=1)


def block_std(hp_b1hw: torch.Tensor, block: int) -> torch.Tensor:
    """Std of the high-pass inside each block x block tile. [B, 1, H//block, W//block]"""
    b, _, h, w = hp_b1hw.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return hp_b1hw.reshape(b, 1, 1, -1).std(dim=-1, keepdim=True)
    t = hp_b1hw[:, :, : hb * block, : wb * block].reshape(b, 1, hb, block, wb, block)
    return t.permute(0, 1, 2, 4, 3, 5).reshape(b, 1, hb, wb, block * block).std(dim=-1)


def texture_mask(std: torch.Tensor, floor: float, target: float) -> torch.Tensor:
    """0..1 weight per block from its texture level.

    ``floor`` > 0: ramps up over [floor, floor + 1]; ``floor`` <= 0 means no floor.
    ``target`` > floor: ramps back down to 0 over [floor, target], so regions
    already at the target texture get nothing; ``target`` <= floor means no target.
    """
    floor = float(floor)
    target = float(target) if target is not None else 0.0
    if floor > 0.0:
        w = ((std - floor) / FLOOR_RAMP).clamp(0.0, 1.0)
    else:
        w = torch.ones_like(std)
        floor = 0.0
    if target > floor:
        w = w * ((target - std) / (target - floor)).clamp(0.0, 1.0)
    return w


def skin_membership(x_bchw: torch.Tensor) -> torch.Tensor:
    """Soft 0..1 skin-tone membership per pixel (YCbCr chroma bands Cb 77-127,
    Cr 133-173, luminance-independent). Measured on Krea 2 portraits: cheeks
    0.86-1.0 for fair, freckled and dark skin; jeans, concrete and a blue wall
    0.0-0.1; sand and hazy sky ~0.3; brown hair and beige fur 0.5-0.9. [B, 1, H, W]"""
    r, g, b = x_bchw[:, 0:1] * 255.0, x_bchw[:, 1:2] * 255.0, x_bchw[:, 2:3] * 255.0
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    def band(v, lo, hi):
        return ((v - lo) / SKIN_SOFT).clamp(0.0, 1.0) * ((hi - v) / SKIN_SOFT).clamp(0.0, 1.0)

    return band(cb, 77.0, 127.0) * band(cr, 133.0, 173.0)


def block_mean(x_b1hw: torch.Tensor, block: int) -> torch.Tensor:
    b, _, h, w = x_b1hw.shape
    hb, wb = h // block, w // block
    if hb == 0 or wb == 0:
        return x_b1hw.mean(dim=(-2, -1), keepdim=True)
    return F.avg_pool2d(x_b1hw[:, :, : hb * block, : wb * block], block)


def tone_fix(out_bchw: torch.Tensor, ref_bchw: torch.Tensor, radius: int = TONE_RADIUS) -> torch.Tensor:
    """Input's low-pass + result's high-pass: restores colour and shading (to a
    fraction of a level; a box blur is not idempotent, so not bit-exact)."""
    return (box(ref_bchw, radius) + (out_bchw - box(out_bchw, radius))).clamp(0.0, 1.0)


def pad_to_multiple(x_bchw: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x_bchw.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph == 0 and pw == 0:
        return x_bchw, (0, 0)
    mode = "reflect" if (ph < h and pw < w) else "replicate"
    return F.pad(x_bchw, (0, pw, 0, ph), mode=mode), (ph, pw)


# -- the operation -----------------------------------------------------------------


def enhance(
    image: torch.Tensor,
    codec: Codec,
    *,
    sigma: float = SIGMA,
    gain: float = GAIN,
    floor: float = FLOOR,
    target: float = TARGET,
    skin_only: bool = True,
    tone: bool = True,
    multiple: int = 16,
    work_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Detail extrapolation through ``codec`` on an image batch.

    image: ``[B, H, W, C]`` float 0..1, C >= 3; channels beyond the first three
    (alpha) pass through untouched. Returns ``(out, mask_preview, stats)`` with
    ``out`` shaped like ``image`` and ``mask_preview`` a 3-channel ``[B, H, W, 3]``
    view of the per-latent-pixel weight, plus one stats dict per image.

    ``skin_only``: multiply the mask by the block-mean skin-tone membership, so
    only skin-coloured regions (and, by colour, brown hair and beige fur) get the
    gain. Off for animals and greyscale images.
    ``multiple``: the codec's pixel stride (16 for Flux.2: 8x latent with a 2x2
    patch). The image is reflect-padded to it before encoding and cropped after.
    ``work_device``: where the blur / mask / tone maths run; the codec moves
    tensors itself. Default: the image's device.
    """
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"expected [B, H, W, C>=3], got {tuple(image.shape)}")
    orig_dtype = image.dtype
    dev = torch.device(work_device) if work_device is not None else image.device
    x_all = image.to(dev).float().permute(0, 3, 1, 2).contiguous()
    x, extra = (x_all[:, :3], x_all[:, 3:]) if x_all.shape[1] > 3 else (x_all, None)
    b, _, h, w = x.shape

    outs, masks, stats = [], [], []
    gain = float(gain)
    for i in range(b):
        xi = x[i : i + 1]
        xp, (ph, pw) = pad_to_multiple(xi, int(multiple))
        hp = highpass_luma_255(xp)

        z = codec.encode(xp.permute(0, 2, 3, 1).contiguous())
        lat_h, lat_w = int(z.shape[-2]), int(z.shape[-1])
        block = max(1, int(round(xp.shape[-2] / lat_h)))

        if gain == 0.0:
            zg = z
            wmap = torch.zeros(1, 1, lat_h, lat_w, device=dev)
        else:
            xb = gaussian_blur(xp, float(sigma))
            zb = codec.encode(xb.permute(0, 2, 3, 1).contiguous())
            std = block_std(hp, block)
            wmap = texture_mask(std, floor, target)
            if skin_only:
                wmap = wmap * block_mean(skin_membership(xp), block)
            wmap = box(wmap, MASK_SMOOTH)
            if wmap.shape[-2:] != (lat_h, lat_w):
                wmap = F.interpolate(wmap, size=(lat_h, lat_w), mode="bilinear", align_corners=False)
            wz = wmap.to(z.device, z.dtype)
            if z.ndim == 5:  # [B, C, T, h, w]
                wz = wz.unsqueeze(2)
            zg = z + gain * wz * (z - zb)

        out = codec.decode(zg)
        out = out.to(dev).float().permute(0, 3, 1, 2)[:, :3]
        if out.shape[-2:] != xp.shape[-2:]:  # a codec that returns a different size: resample rather than crash
            out = F.interpolate(out, size=xp.shape[-2:], mode="bilinear", align_corners=False)
        if tone:
            out = tone_fix(out, xp)
        out = out[:, :, : xp.shape[-2] - ph, : xp.shape[-1] - pw].clamp(0.0, 1.0)

        mask_full = F.interpolate(wmap.to(dev), size=xp.shape[-2:], mode="nearest")[:, :, : h, : w]
        hf_in = hp[:, :, : h, : w].reshape(1, -1).std(dim=1)
        hf_out = highpass_luma_255(out).reshape(1, -1).std(dim=1)
        stats.append(
            {
                "hf_in_255": hf_in.item(),
                "hf_out_255": hf_out.item(),
                "mask_mean": wmap.mean().item(),
                "mask_cover_pct": (wmap > 0.5).float().mean().item() * 100.0,
                "latent": f"{int(z.shape[1])}x{lat_h}x{lat_w}",
                "block_px": block,
                "gain": gain,
                "sigma": float(sigma),
                "floor": float(floor),
                "target": float(target),
                "skin_only": bool(skin_only),
                "tone": bool(tone),
            }
        )
        outs.append(out)
        masks.append(mask_full)

    out = torch.cat(outs, dim=0)
    mask = torch.cat(masks, dim=0).expand(-1, 3, -1, -1)
    if extra is not None:
        out = torch.cat([out, extra], dim=1)
    out = out.permute(0, 2, 3, 1).contiguous().to(image.device, orig_dtype)
    mask = mask.permute(0, 2, 3, 1).contiguous().to(image.device, orig_dtype)
    return out, mask, stats


def describe_codec(codec) -> tuple[str, bool]:
    """('flux2', True) for the architecture this was measured on, else (name, False)."""
    channels = getattr(codec, "latent_channels", None)
    ratio = getattr(codec, "downscale_ratio", None)
    if channels == 128 and ratio == 16:
        return "flux2", True
    inner = getattr(codec, "first_stage_model", None)
    name = type(inner).__name__ if inner is not None else type(codec).__name__
    return f"{name} ({channels} ch, {ratio}x)", False


def status_line(
    stats: list,
    *,
    grid_in_255: float | None = None,
    grid_in_removed: bool | None = None,
    grid_out_255: float | None = None,
    seconds: float | None = None,
) -> str:
    """One-line verdict shared by the front ends. No colons (Forge quotes them)."""
    s = stats[0]
    hf_in, hf_out = s["hf_in_255"], s["hf_out_255"]
    pct = (hf_out / hf_in - 1.0) * 100.0 if hf_in > 1e-6 else 0.0
    mask_txt = f"mask {s['mask_cover_pct']:.0f}%"
    if s.get("skin_only"):
        mask_txt += " (skin only)"
    parts = [
        f"texture {hf_in:.2f} -> {hf_out:.2f} /255 ({pct:+.0f}%)",
        mask_txt,
        f"gain {s['gain']:g} sigma {s['sigma']:g}",
    ]
    if grid_in_255 is not None:
        if grid_in_removed:
            parts.append(f"input grid {grid_in_255:.2f}/255 removed first")
        else:
            parts.append(f"input grid {grid_in_255:.2f}/255 (clean)")
    if grid_out_255 is not None:
        parts.append(f"output grid {grid_out_255:.2f}/255")
    if seconds is not None:
        parts.append(f"{seconds:.1f} s")
    line = " · ".join(parts)
    if len(stats) > 1:
        line += f" · batch of {len(stats)} (first shown)"
    return line
