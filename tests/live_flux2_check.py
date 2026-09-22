"""Live check of the enhancement through the REAL Flux.2 VAE, using a Forge Neo checkout.

Not picked up by ``unittest discover`` (no ``test_`` prefix): it needs the Forge venv,
a GPU and a Flux.2 VAE file in the checkout's ``models/VAE``. No checkpoint is needed.

    <forge>/venv/Scripts/python.exe tests/live_flux2_check.py [--forge PATH] [--image PNG] [--out DIR]

What it verifies:
  1. the loader finds the Flux.2 VAE by header and builds it with correctly mapped weights
     (a plain round trip must reconstruct a synthetic image above 30 dB; random weights give ~14);
  2. gain 0.5 raises the texture of a mid-texture band and leaves a flat band alone;
  3. the whole Forge pipeline (notch -> enhance) removes a synthetic 2px grid;
  4. optionally, the pipeline on a real image, written to --out for eyeballing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", default="T:/claude/github/sd-webui-forge-classic")
    ap.add_argument("--image", default=None, help="optional PNG to run the full pipeline on")
    ap.add_argument("--out", default=None, help="where to write the enhanced image / mask (default: next to --image)")
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=None, help="mask floor for the real image (default: the core's)")
    ap.add_argument("--target", type=float, default=None, help="texture target for the real image (default: the core's)")
    ap.add_argument("--skin", type=int, default=1, help="1 = skin tones only (portraits), 0 = everything (animals)")
    args = ap.parse_args()
    sys.argv = sys.argv[:1]  # Forge's backend.args parses sys.argv at import and rejects ours

    forge = os.path.abspath(args.forge)
    sys.path.insert(0, os.path.join(forge, "modules_forge", "packages"))
    sys.path.insert(0, forge)
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(forge)

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(REPO_ROOT / "tests"))
    import forge_stubs as stubs  # the synthetic images only; nothing is stubbed into sys.modules

    from lib_degrid import flux2_vae
    from lib_degrid.loader import load_core, load_enhance_core

    core = load_core(str(REPO_ROOT))
    en = load_enhance_core(str(REPO_ROOT))

    vae_dir = os.path.join(forge, "models", "VAE")
    vae_dict = {os.path.splitext(f)[0]: os.path.join(vae_dir, f) for f in os.listdir(vae_dir) if f.lower().endswith(".safetensors")}
    names = flux2_vae.candidates(vae_dict)
    print(f"VAE folder {vae_dir}: {len(vae_dict)} safetensors, Flux.2 candidates {names}")
    if not names:
        print("FAIL: no Flux.2 VAE found by header")
        return 1
    t0 = time.perf_counter()
    vae = flux2_vae.get(vae_dict[names[0]])
    print(f"built {names[0]} in {time.perf_counter() - t0:.1f}s; describe_codec -> {en.describe_codec(vae)}")
    failures = 0

    # 1. round trip fidelity on a synthetic image (in 0.25..0.75 so clamping plays no part)
    g = torch.Generator().manual_seed(0)
    h, w = 256, 320
    base = stubs.smooth_image(h, w, seed=1) * 0.5 + 0.25
    noise = torch.randn(1, 3, h, w, generator=g)
    sigma = torch.zeros(1, 1, 1, w)
    sigma[..., w // 2 :] = 5.0 / 255.0
    x = (base + noise * sigma).clamp(0, 1).permute(0, 2, 3, 1).contiguous()
    with torch.inference_mode():
        rt, _, _ = en.enhance(x, vae, gain=0.0, tone=False, work_device=vae.device)
        psnr = 10 * torch.log10(1.0 / ((rt.float() - x) ** 2).mean()).item()
        print(f"1. round trip PSNR {psnr:.2f} dB (expect > 30; ~14 means the weights did not map)")
        if psnr < 30:
            failures += 1

        # 2. gain 0.5 texture response per band
        out, mask, st = en.enhance(x, vae, gain=args.gain, skin_only=False, floor=3.0, target=9.0, tone=True, work_device=vae.device)
        hp_in = en.highpass_luma_255(x.permute(0, 3, 1, 2))
        hp_out = en.highpass_luma_255(out.float().permute(0, 3, 1, 2))
        flat_in, flat_out = hp_in[..., : w // 2 - 16].std().item(), hp_out[..., : w // 2 - 16].std().item()
        tex_in, tex_out = hp_in[..., w // 2 + 16 :].std().item(), hp_out[..., w // 2 + 16 :].std().item()
        print(f"2. gain {args.gain:g}: flat band {flat_in:.2f} -> {flat_out:.2f}, textured band {tex_in:.2f} -> {tex_out:.2f} (/255); {en.status_line(st)}")
        if not (tex_out > tex_in * 1.1):
            print("   FAIL: the textured band did not gain texture")
            failures += 1
        if flat_out > max(1.0, flat_in * 1.5):
            print("   FAIL: the flat band gained texture")
            failures += 1

        # 3. the pipeline on a gridded synthetic image
        gridded = stubs.add_lattice(x.permute(0, 3, 1, 2), 1.5, 1.2).permute(0, 2, 3, 1).contiguous()
        cleaned, _, dg = core.degrid(gridded.to(vae.device), mode="auto")
        out3, _, st3 = en.enhance(cleaned, vae, gain=args.gain, skin_only=False, work_device=vae.device)
        grid_out = core.lattice_amp(core.extract_grid(out3[..., :3].permute(0, 3, 1, 2).float()))[0][0].item() * 255
        print(f"3. pipeline: input grid {dg[0]['amp_255']:.2f}/255 -> output grid {grid_out:.2f}/255")
        if grid_out > 0.5:
            print("   FAIL: grid survived the pipeline")
            failures += 1

        # 4. optional real image
        if args.image:
            img = Image.open(args.image).convert("RGB")
            xi = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0)[None].to(vae.device)
            t0 = time.perf_counter()
            cleaned, _, dg = core.degrid(xi, mode="auto")
            kw = {"floor": args.floor} if args.floor is not None else {}
            if args.target is not None:
                kw["target"] = args.target
            out4, mask4, st4 = en.enhance(cleaned, vae, gain=args.gain, skin_only=bool(args.skin), work_device=vae.device, **kw)
            grid_out = core.lattice_amp(core.extract_grid(out4[..., :3].permute(0, 3, 1, 2).float()))[0][0].item() * 255
            line = en.status_line(st4, grid_in_255=dg[0]["amp_255"], grid_in_removed=not dg[0]["skipped"], grid_out_255=grid_out, seconds=time.perf_counter() - t0)
            print(f"4. {os.path.basename(args.image)} {img.size}: {line}")
            out_dir = args.out or os.path.dirname(os.path.abspath(args.image))
            os.makedirs(out_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(args.image))[0]
            for name, t in (("enhanced", out4), ("mask", mask4)):
                arr = (t[0].clamp(0, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
                Image.fromarray(arr).save(os.path.join(out_dir, f"{stem}_{name}_g{args.gain:g}_s{args.skin}.png"))
            print(f"   written to {out_dir}")

    print("PASS" if failures == 0 else f"{failures} FAILURE(S)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
