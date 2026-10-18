"""Offline tests for the VAE enhancement (core + Forge script + Flux.2 loader logic).

    python -m unittest discover -s tests -v

Needs torch, numpy and Pillow; no Forge checkout, no gradio, no VAE file. The
codec under test is an identity pixel-(un)shuffle with the Flux.2 geometry, so
the latent-space maths has an exact pixel-space equivalent to compare against.
The real VAE is exercised by ``tests/live_flux2_check.py`` (needs the Forge venv).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_stubs as stubs  # noqa: E402

stubs.install_fakes()

from lib_degrid import flux2_vae  # noqa: E402
from lib_degrid.loader import load_core, load_enhance_core  # noqa: E402

core = load_core(str(stubs.REPO_ROOT))
en = load_enhance_core(str(stubs.REPO_ROOT))
SCRIPT_PATH = stubs.REPO_ROOT / "scripts" / "vae_enhance_forge.py"
script_mod = stubs.load_script_module(SCRIPT_PATH, "vae_enhance_forge_under_test")

H, W = 96, 128


def lattice_255(x_bhwc: torch.Tensor) -> float:
    corr = core.extract_grid(x_bhwc[..., :3].permute(0, 3, 1, 2).float())
    return core.lattice_amp(corr)[0].max().item() * 255.0


def banded_image(h: int = H, w: int = W, seed: int = 0) -> torch.Tensor:
    """[1, 3, h, w] in 0.25..0.75: three vertical bands of texture, flat / std ~5 / std ~15 (/255)."""
    g = torch.Generator().manual_seed(seed)
    base = stubs.smooth_image(h, w, seed=seed) * 0.5 + 0.25
    noise = torch.randn(1, 1, h, w, generator=g).expand(1, 3, h, w)  # same noise in every channel: luma std == sigma
    sigma = torch.zeros(1, 1, 1, w)
    sigma[..., w // 3 : 2 * w // 3] = 5.0 / 255.0
    sigma[..., 2 * w // 3 :] = 15.0 / 255.0
    return (base + noise * sigma).clamp(0.0, 1.0)


def pixel_unsharp(x_bchw: torch.Tensor, sigma: float, gain: float) -> torch.Tensor:
    return x_bchw + gain * (x_bchw - en.gaussian_blur(x_bchw, sigma))


def write_fake_safetensors(path: Path, tensors: dict[str, list[int]]) -> None:
    header, offset = {}, 0
    for name, shape in tensors.items():
        n = 4
        for s in shape:
            n *= s
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * min(offset, 64))  # truncated body: only the header is ever read


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.codec = stubs.FakeFlux2VAE()

    def test_identity_codec_equals_pixel_unsharp(self):
        x = banded_image().permute(0, 2, 3, 1)
        out, mask, st = en.enhance(x, self.codec, sigma=1.0, gain=0.5, floor=0.0, target=0.0, skin_only=False, tone=False)
        expected = pixel_unsharp(x.permute(0, 3, 1, 2), 1.0, 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.allclose(out, expected, atol=1e-5), (out - expected).abs().max())
        self.assertTrue(torch.all(mask == 1.0))  # floor 0 -> every block gets full gain
        self.assertEqual(st[0]["block_px"], 16)
        self.assertEqual(st[0]["latent"], "768x6x8")
        self.assertGreater(st[0]["hf_out_255"], st[0]["hf_in_255"])
        self.assertEqual(self.codec.encode_calls, 2)
        self.assertEqual(self.codec.decode_calls, 1)

    def test_gain_zero_is_a_round_trip(self):
        x = banded_image().permute(0, 2, 3, 1)
        out, mask, st = en.enhance(x, self.codec, gain=0.0, tone=False)
        self.assertTrue(torch.allclose(out, x, atol=1e-6))
        self.assertTrue(torch.all(mask == 0.0))
        self.assertEqual(self.codec.encode_calls, 1)  # no blurred encode when there is nothing to extrapolate

    def test_tone_fix_restores_the_input_low_pass(self):
        # a decoder that shifts colour, like the real one does (red down, green/blue up)
        cast = torch.tensor([-0.02, 0.006, 0.01]).view(1, 1, 1, 3)
        codec = self.codec
        plain_decode = codec.decode
        codec.decode = lambda z: plain_decode(z) + cast
        x = banded_image().permute(0, 2, 3, 1)
        out, _, _ = en.enhance(x, codec, skin_only=False, sigma=1.0, gain=1.5, floor=0.0, target=0.0, tone=True)
        raw, _, _ = en.enhance(x, codec, skin_only=False, sigma=1.0, gain=1.5, floor=0.0, target=0.0, tone=False)
        lo_in = en.box(x.permute(0, 3, 1, 2), en.TONE_RADIUS)
        lo_out = en.box(out.permute(0, 3, 1, 2), en.TONE_RADIUS)
        lo_raw = en.box(raw.permute(0, 3, 1, 2), en.TONE_RADIUS)
        self.assertGreater((lo_in - lo_raw).abs().mean().item(), 0.01)  # the cast is there without the fix
        self.assertLess((lo_in - lo_out).abs().mean().item(), 1e-3)  # and gone with it (a box blur is not idempotent, so not bit-exact)
        # ...while the fine detail is still the extrapolated one
        self.assertGreater(en.hf_energy_255(out.permute(0, 3, 1, 2)).item(), en.hf_energy_255(x.permute(0, 3, 1, 2)).item() * 1.3)

    def test_mask_floor_and_target(self):
        w = 288  # 96px bands = 6 latent blocks each, so a 2-block margin on each side leaves a 2-block interior
        x = banded_image(H, w).permute(0, 2, 3, 1)
        _, mask, st = en.enhance(x, self.codec, skin_only=False, gain=0.5, floor=3.0, target=9.0)
        m = mask[0, :, :, 0]
        flat = m[:, : w // 3 - 32].mean().item()  # interior of each band: border blocks mix textures and the 3x3 smoothing spreads them one block
        mid = m[:, w // 3 + 32 : 2 * w // 3 - 32].mean().item()
        strong = m[:, 2 * w // 3 + 32 :].mean().item()
        self.assertLess(flat, 0.05)
        self.assertGreater(mid, 0.45)
        self.assertLess(mid, 0.9)  # std 5 sits at (9-5)/(9-3) = 0.67 of the target ramp
        self.assertLess(strong, 0.05)
        # target off: the strong band gets full gain, the flat band still nothing
        _, mask2, _ = en.enhance(x, self.codec, skin_only=False, gain=0.5, floor=3.0, target=0.0)
        self.assertGreater(mask2[0, :, 2 * w // 3 + 32 :, 0].mean().item(), 0.95)
        self.assertLess(mask2[0, :, : w // 3 - 32, 0].mean().item(), 0.05)
        # floor off: the flat band gets full gain too
        _, mask3, _ = en.enhance(x, self.codec, skin_only=False, gain=0.5, floor=0.0, target=0.0)
        self.assertTrue(torch.all(mask3 == 1.0))
        self.assertGreater(st[0]["mask_cover_pct"], 10.0)
        self.assertLess(st[0]["mask_cover_pct"], 60.0)

    def test_skin_only_mask(self):
        w = 288
        grey = banded_image(H, w)  # neutral: no skin tone anywhere
        skin = grey * torch.tensor([1.0, 0.78, 0.68]).view(1, 3, 1, 1)  # a skin-like tint
        self.assertLess(en.skin_membership(grey).mean().item(), 0.05)
        self.assertGreater(en.skin_membership(skin).mean().item(), 0.7)  # the darkest and brightest blocks fall off the soft bands
        x_grey = grey.permute(0, 2, 3, 1)
        x_skin = skin.permute(0, 2, 3, 1)
        _, m_grey_on, st = en.enhance(x_grey, self.codec, gain=0.5, floor=0.0, target=0.0, skin_only=True)
        _, m_grey_off, _ = en.enhance(x_grey, self.codec, gain=0.5, floor=0.0, target=0.0, skin_only=False)
        _, m_skin_on, _ = en.enhance(x_skin, self.codec, gain=0.5, floor=0.0, target=0.0, skin_only=True)
        self.assertLess(m_grey_on.max().item(), 0.05)  # grey image, skin only: nothing gets the gain
        self.assertTrue(torch.all(m_grey_off == 1.0))
        self.assertGreater(m_skin_on.mean().item(), 0.6)
        self.assertTrue(st[0]["skin_only"])
        self.assertIn("(skin only)", en.status_line(st))

    def test_odd_size_and_rgba_passthrough(self):
        rgb = banded_image(70, 100)
        alpha = stubs.smooth_image(70, 100, c=1, seed=5)
        x = torch.cat([rgb, alpha], dim=1).permute(0, 2, 3, 1)
        out, mask, st = en.enhance(x, self.codec, skin_only=False, sigma=1.0, gain=0.5, floor=0.0, target=0.0, tone=False)
        self.assertEqual(tuple(out.shape), (1, 70, 100, 4))
        self.assertEqual(tuple(mask.shape), (1, 70, 100, 3))
        self.assertTrue(torch.equal(out[..., 3], x[..., 3]))
        # away from the padded edges the maths is the unpadded pixel unsharp
        expected = pixel_unsharp(rgb, 1.0, 0.5).clamp(0, 1).permute(0, 2, 3, 1)
        self.assertTrue(torch.allclose(out[:, 8:-8, 8:-8, :3], expected[:, 8:-8, 8:-8], atol=1e-5))
        self.assertEqual(st[0]["latent"], "768x5x7")  # padded to 80x112

    def test_batch_and_dtype(self):
        x = torch.cat([banded_image(seed=1), banded_image(seed=2)], dim=0).permute(0, 2, 3, 1).half()
        out, mask, st = en.enhance(x, self.codec, skin_only=False, gain=0.5)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(len(st), 2)
        self.assertEqual(tuple(out.shape), tuple(x.shape))

    def test_describe_and_status_line(self):
        self.assertEqual(en.describe_codec(self.codec), ("flux2", True))

        class Other:
            latent_channels = 16
            downscale_ratio = 8

        name, ok = en.describe_codec(Other())
        self.assertFalse(ok)
        self.assertIn("16 ch", name)
        st = [{"hf_in_255": 4.0, "hf_out_255": 5.0, "mask_cover_pct": 41.2, "gain": 0.5, "sigma": 1.0}]
        line = en.status_line(st, grid_in_255=1.25, grid_in_removed=True, grid_out_255=0.04, seconds=2.6)
        self.assertEqual(line, "texture 4.00 -> 5.00 /255 (+25%) · mask 41% · gain 0.5 sigma 1 · input grid 1.25/255 removed first · output grid 0.04/255 · 2.6 s")
        self.assertNotIn(":", line)
        line = en.status_line(st * 2, grid_in_255=0.1, grid_in_removed=False)
        self.assertIn("input grid 0.10/255 (clean)", line)
        self.assertIn("batch of 2", line)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.flux2 = d / "flux2-vae.safetensors"
        write_fake_safetensors(self.flux2, {"decoder.conv_in.weight": [512, 32, 3, 3], "decoder.up_blocks.0.resnets.0.norm1.weight": [512], "bn.running_mean": [128], "bn.running_var": [128]})
        self.flux1 = d / "ae.safetensors"
        write_fake_safetensors(self.flux1, {"decoder.conv_in.weight": [512, 16, 3, 3]})
        self.wan = d / "wan_2.1_vae.safetensors"
        write_fake_safetensors(self.wan, {"decoder.middle.0.residual.0.gamma": [384], "conv2.weight": [16, 32, 1, 1, 1]})
        self.nobn = d / "odd.safetensors"
        write_fake_safetensors(self.nobn, {"decoder.conv_in.weight": [512, 32, 3, 3]})
        flux2_vae._header_cache.clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_detect_by_header(self):
        self.assertEqual(flux2_vae.detect_arch(flux2_vae.read_header(self.flux2)), "flux2")
        self.assertIsNone(flux2_vae.detect_arch(flux2_vae.read_header(self.flux1)))
        self.assertIsNone(flux2_vae.detect_arch(flux2_vae.read_header(self.wan)))
        self.assertIsNone(flux2_vae.detect_arch(flux2_vae.read_header(self.nobn)))
        self.assertEqual(flux2_vae.read_header(Path(self.tmp.name) / "missing.safetensors"), {})
        self.assertIsNone(flux2_vae.detect_arch({}))

    def test_candidates_filter_and_cache(self):
        vae_dict = {"flux2-vae": str(self.flux2), "ae": str(self.flux1), "wan": str(self.wan), "odd": str(self.nobn), "gone": str(Path(self.tmp.name) / "gone.safetensors")}
        self.assertEqual(flux2_vae.candidates(vae_dict), ["flux2-vae"])
        self.assertEqual(len(flux2_vae._header_cache), 4)  # the missing file is not cached
        # a rewritten file is re-read
        write_fake_safetensors(self.flux1, {"decoder.conv_in.weight": [512, 32, 3, 3], "bn.running_mean": [128]})
        os.utime(self.flux1, (0, 1))  # force a different mtime regardless of clock resolution
        self.assertEqual(flux2_vae.candidates(vae_dict), ["ae", "flux2-vae"])
        self.assertEqual(flux2_vae.candidates({}), [])


class ScriptTests(unittest.TestCase):
    UI_ORDER = ["enabled", "vae_name", "gain", "sigma", "floor", "target", "skin_only", "tone_fix", "post_notch", "show_mask"]
    DEFAULTS = dict(enabled=False, vae_name="flux2-vae", gain=0.5, sigma=1.0, floor=0.5, target=9.0, skin_only=False, tone_fix=True, post_notch=False, show_mask=False)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "flux2-vae.safetensors"
        write_fake_safetensors(path, {"decoder.conv_in.weight": [512, 32, 3, 3], "bn.running_mean": [128]})
        sd_vae = sys.modules["modules.sd_vae"]
        sd_vae.vae_dict.clear()
        sd_vae.vae_dict["flux2-vae"] = str(path)
        sd_vae.vae_dict["ae"] = str(Path(self.tmp.name) / "ae.safetensors")
        write_fake_safetensors(Path(sd_vae.vae_dict["ae"]), {"decoder.conv_in.weight": [512, 16, 3, 3]})
        flux2_vae._header_cache.clear()
        self.codec = stubs.FakeFlux2VAE()
        self._orig_get = script_mod.flux2_vae.get
        script_mod.flux2_vae.get = lambda p: self.codec
        self.script = script_mod.VAEEnhanceScript()
        script_mod.VAEEnhanceScript._previews = []
        script_mod.VAEEnhanceScript._warned = set()

    def tearDown(self):
        script_mod.flux2_vae.get = self._orig_get
        self.tmp.cleanup()

    def args(self, **overrides):
        values = dict(self.DEFAULTS, **overrides)
        return [values[name] for name in self.UI_ORDER]

    @staticmethod
    def gridded_pil() -> Image.Image:
        img = stubs.add_lattice(banded_image(), 1.5, 1.0)  # [1, 3, H, W]
        arr = img[0].permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
        return Image.fromarray(arr, "RGB")

    def test_ui_order_matches_hooks_and_defaults(self):
        controls = self.script.ui(False)
        for hook in (self.script.process, self.script.postprocess_image_after_composite):
            params = [n for n in inspect.signature(hook).parameters if n not in ("p", "pp", "kwargs")]
            self.assertEqual(params, self.UI_ORDER, hook.__name__)
        self.assertEqual(len(controls), len(self.UI_ORDER))
        ui_defaults = dict(self.DEFAULTS, skin_only=True)  # the UI default is on; the tests' synthetic images are grey, so they run with it off
        for control, name in zip(controls, self.UI_ORDER):
            self.assertEqual(control.value, ui_defaults[name], name)
        self.assertEqual(controls[1].choices, ["flux2-vae"])  # only the Flux.2 file is offered
        self.assertEqual(len(self.script.infotext_fields), 9)

    def test_no_flux2_vae_lists_none(self):
        sys.modules["modules.sd_vae"].vae_dict.pop("flux2-vae")
        controls = self.script.ui(False)
        self.assertEqual(controls[1].choices, [script_mod.NONE])
        self.assertEqual(controls[1].value, script_mod.NONE)

    def test_disabled_and_missing_vae_are_no_ops(self):
        p = stubs.FakeProcessing(stubs.FakeVAE(stubs.FakeDecoder(torch.zeros(1, 3, 1, H, W))))
        image = self.gridded_pil()
        pp = stubs.fake_forge_modules(str(stubs.REPO_ROOT))["modules.scripts"].PostprocessImageArgs(image, 0)
        self.script.process(p, *self.args(enabled=False))
        self.script.postprocess_image_after_composite(p, pp, *self.args(enabled=False))
        self.assertIs(pp.image, image)
        self.assertNotIn(script_mod.INFOTEXT_KEY, p.extra_generation_params)
        with self.assertLogs("processing", level="ERROR"):
            self.script.process(p, *self.args(enabled=True, vae_name="does-not-exist"))
        self.script.postprocess_image_after_composite(p, pp, *self.args(enabled=True, vae_name="does-not-exist"))
        self.assertIs(pp.image, image)
        self.assertNotIn(script_mod.INFOTEXT_KEY, p.extra_generation_params)

    def test_full_run_notches_first_and_enhances(self):
        p = stubs.FakeProcessing(stubs.FakeVAE(stubs.FakeDecoder(torch.zeros(1, 3, 1, H, W))))
        image = self.gridded_pil()
        x_in = torch.from_numpy(__import__("numpy").asarray(image, dtype="float32") / 255.0)[None]
        self.assertGreater(lattice_255(x_in), 2.0)
        pp = stubs.fake_forge_modules(str(stubs.REPO_ROOT))["modules.scripts"].PostprocessImageArgs(image, 1)

        self.script.process(p, *self.args(enabled=True, show_mask=True))
        self.assertEqual(p.extra_generation_params[script_mod.INFOTEXT_KEY], "vae=flux2-vae;gain=0.5;sigma=1;floor=0.5;target=9;skin=0;tonefix=1;postnotch=0")
        with self.assertLogs("processing", level="INFO") as logs:
            self.script.postprocess_image_after_composite(p, pp, *self.args(enabled=True, show_mask=True))
        self.assertEqual(len(logs.output), 1)
        self.assertIn(f"VAE Enhance: {W}x{H} [2]", logs.output[0])
        self.assertTrue(logs.output[0].isascii(), logs.output[0])
        self.assertIn("input grid", logs.output[0])
        self.assertIn("removed first", logs.output[0])

        self.assertIsNot(pp.image, image)
        self.assertEqual(pp.image.size, image.size)
        x_out = torch.from_numpy(__import__("numpy").asarray(pp.image, dtype="float32") / 255.0)[None]
        self.assertLess(lattice_255(x_out), 0.4)  # the identity codec cannot remove a grid: the pre-notch did
        notched, _, _ = core.degrid(x_in, mode="auto")
        band = slice(W // 3 + 16, 2 * W // 3 - 16)  # the mid-texture band is the one the mask lets through
        hf_notched = en.highpass_luma_255(notched.permute(0, 3, 1, 2))[..., band].std().item()
        hf_out = en.highpass_luma_255(x_out.permute(0, 3, 1, 2))[..., band].std().item()
        self.assertGreater(hf_out, hf_notched * 1.1)  # gained texture on top of the notched image
        flat = slice(0, W // 3 - 16)
        self.assertLess(en.highpass_luma_255(x_out.permute(0, 3, 1, 2))[..., flat].std().item(), en.highpass_luma_255(notched.permute(0, 3, 1, 2))[..., flat].std().item() * 1.2)
        result = p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY]
        self.assertTrue(result.startswith("texture "))
        self.assertTrue(result.isascii())
        self.assertNotIn(":", result)
        self.assertEqual(len(script_mod.VAEEnhanceScript._previews), 1)

        processed = stubs.FakeProcessed()
        self.script.postprocess(p, processed)
        self.assertEqual(len(processed.extra_images), 1)
        self.assertEqual(processed.extra_images[0].size, image.size)
        self.assertEqual(processed.infotexts[1], processed.infotexts[0])
        self.assertEqual(script_mod.VAEEnhanceScript._previews, [])

    def test_post_notch_and_untested_vae_flag(self):
        p = stubs.FakeProcessing(stubs.FakeVAE(stubs.FakeDecoder(torch.zeros(1, 3, 1, H, W))))
        pp = stubs.fake_forge_modules(str(stubs.REPO_ROOT))["modules.scripts"].PostprocessImageArgs(self.gridded_pil(), 0)
        self.codec.latent_channels = 16  # pretend it is some other VAE
        self.script.process(p, *self.args(enabled=True, post_notch=True))
        with self.assertLogs("processing", level="INFO") as logs:
            self.script.postprocess_image_after_composite(p, pp, *self.args(enabled=True, post_notch=True))
        self.assertIn("untested VAE", logs.output[0])
        self.assertIn("postnotch=1", p.extra_generation_params[script_mod.INFOTEXT_KEY])

    def test_paste_fields(self):
        self.script.ui(False)
        names = ["enabled", "vae", "gain", "sigma", "floor", "target", "skin", "tonefix", "postnotch"]
        readers = {name: pf.function for pf, name in zip(self.script.infotext_fields, names)}
        params = {script_mod.INFOTEXT_KEY: "vae=flux2-vae;gain=0.75;sigma=1.2;floor=2.5;target=0;skin=0;tonefix=0;postnotch=1"}
        self.assertIs(readers["enabled"](params), True)
        self.assertEqual(readers["vae"](params), "flux2-vae")
        self.assertAlmostEqual(readers["gain"](params), 0.75)
        self.assertAlmostEqual(readers["sigma"](params), 1.2)
        self.assertAlmostEqual(readers["floor"](params), 2.5)
        self.assertAlmostEqual(readers["target"](params), 0.0)
        self.assertIs(readers["skin"](params), False)
        self.assertIs(readers["tonefix"](params), False)
        self.assertIs(readers["postnotch"](params), True)
        self.assertIs(readers["enabled"]({}), False)
        self.assertIsNone(readers["gain"]({}))
        self.assertIsNone(script_mod.parse_infotext("garbage"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
