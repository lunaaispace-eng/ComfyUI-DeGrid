"""Offline tests for the Forge Neo extension. Run from the repo root:

    python -m unittest discover -s tests -v

Needs torch, numpy and Pillow (any Forge / ComfyUI venv has them); no Forge
checkout, no gradio, no checkpoint.
"""

from __future__ import annotations

import inspect
import logging
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_stubs as stubs  # noqa: E402

stubs.install_fakes()

from lib_degrid import vae_wrapper as vw  # noqa: E402
from lib_degrid.loader import load_core  # noqa: E402

core = load_core(str(stubs.REPO_ROOT))
SCRIPT_PATH = stubs.REPO_ROOT / "scripts" / "degrid_forge.py"
script_mod = stubs.load_script_module(SCRIPT_PATH, "degrid_forge_under_test")

H, W = 96, 128


def lattice_255(x_bhwc: torch.Tensor) -> float:
    corr = core.extract_grid(x_bhwc.permute(0, 3, 1, 2).float())
    return core.lattice_amp(corr)[0].max().item() * 255.0


def gridded_5d() -> torch.Tensor:
    img = stubs.add_lattice(stubs.smooth_image(H, W))  # [1, 3, H, W]
    return img.unsqueeze(2)  # [1, C, T, H, W]; FakeDecoder expands to the latent's batch


def clean_5d() -> torch.Tensor:
    return stubs.smooth_image(H, W, seed=3).unsqueeze(2)


class CoreTests(unittest.TestCase):
    def test_threshold_default_matches_constant(self):
        x = stubs.add_lattice(stubs.smooth_image(H, W)).permute(0, 2, 3, 1)
        _, _, a = core.degrid(x)
        _, _, b = core.degrid(x, threshold=core.NEGLIGIBLE_AMP)
        self.assertEqual(a[0]["skipped"], b[0]["skipped"])
        self.assertFalse(a[0]["skipped"])
        _, _, c = core.degrid(x, threshold=10.0 / 255.0)
        self.assertTrue(c[0]["skipped"])

    def test_status_line_text_matches_node(self):
        removed = {"amp_255": 2.1, "limit": 0.019, "skipped": False, "clipped_pct": 1.2,
                   "checker_255": 1.9, "vstripe_255": 0.3, "hstripe_255": 0.2}
        self.assertEqual(
            core.status_line("auto", [removed]),
            "grid 2.10/255 (checker) — removed (limit 0.019 auto) · edges protected 1.2%",
        )
        clean = {"amp_255": 0.16, "limit": 0.004, "skipped": True, "clipped_pct": 0.0,
                 "checker_255": 0.1, "vstripe_255": 0.05, "hstripe_255": 0.05}
        self.assertEqual(
            core.status_line("manual", [clean, clean]),
            "grid 0.16/255 — none detected, passed through untouched · edges protected 0.0% · batch of 2 (first shown)",
        )
        # a raised threshold moves the "none detected" line with it
        self.assertIn("none detected", core.status_line("auto", [removed], threshold=3.0 / 255.0))
        # a starved clamp is reported as partial, with what is left
        partial = dict(removed, residual_255=1.66, limit=0.001)
        line = core.status_line("manual", [partial])
        self.assertIn("partially removed, 1.66/255 left (limit 0.001 manual too low)", line)
        self.assertNotIn(":", line)

    def test_rgba_filters_colour_and_passes_alpha_through(self):
        rgb = stubs.add_lattice(stubs.smooth_image(H, W))  # [1, 3, H, W]
        alpha = stubs.add_lattice(stubs.smooth_image(H, W, c=1, seed=7), 2.0, 2.0)  # gridded alpha, must survive
        rgba = torch.cat([rgb, alpha], dim=1).permute(0, 2, 3, 1)
        cleaned, vis, st = core.degrid(rgba, grid_view="full frame")
        self.assertEqual(tuple(cleaned.shape), (1, H, W, 4))
        self.assertTrue(torch.equal(cleaned[..., 3], rgba[..., 3]))
        self.assertLess(lattice_255(cleaned[..., :3]), 0.3)
        self.assertTrue(torch.all(vis[..., 3] == 1.0))
        # the alpha lattice must not leak into the measurement
        _, _, st_rgb = core.degrid(rgba[..., :3])
        self.assertAlmostEqual(st[0]["amp_255"], st_rgb[0]["amp_255"], places=5)

    def test_auto_raises_the_limit_until_the_grid_is_gone(self):
        # A grid that is faint over most of the frame but strong in one band: the
        # 75th-percentile guess fits the faint part and clamps the band away.
        base = stubs.smooth_image(H, W)
        faint = stubs.add_lattice(base, 0.5, 0.5)
        strong = stubs.add_lattice(base, 6.0, 6.0)
        img = faint.clone()
        img[:, :, : H // 4, :] = strong[:, :, : H // 4, :]
        x = img.permute(0, 2, 3, 1)
        corr = core.extract_grid(img)
        guess = core.auto_limit(corr)[0].item()
        chosen = core.auto_limit_targeted(corr)[0].item()
        self.assertGreater(core.residual_after_clamp(corr, torch.tensor([guess]))[0].item() * 255, 0.5)
        self.assertGreater(chosen, guess)
        self.assertLessEqual(chosen, 0.05 + 1e-6)  # float32 ceiling
        cleaned, _, st = core.degrid(x, mode="auto")
        self.assertAlmostEqual(st[0]["limit"], chosen, places=6)
        self.assertLess(st[0]["residual_255"], 0.25 + 1e-6)
        self.assertLess(lattice_255(cleaned), 0.3)
        self.assertIn("\u2014 removed", core.status_line("auto", st))
        # the residual measurement is exact: it matches a direct measurement of the output
        self.assertAlmostEqual(st[0]["residual_255"], lattice_255(cleaned), delta=0.05)
        # an easy image stops at the first rung
        easy = stubs.add_lattice(base).permute(0, 2, 3, 1)
        _, _, st_easy = core.degrid(easy, mode="auto")
        self.assertAlmostEqual(st_easy[0]["limit"], core.auto_limit(core.extract_grid(easy.permute(0, 3, 1, 2)))[0].item(), places=6)
        # batch: each image gets its own rung
        both = torch.cat([x, easy], dim=0)
        _, _, st_both = core.degrid(both, mode="auto")
        self.assertAlmostEqual(st_both[0]["limit"], chosen, places=6)
        self.assertAlmostEqual(st_both[1]["limit"], st_easy[0]["limit"], places=6)

    def test_local_limits_follow_the_grid(self):
        # 512x512, tiles of 128: a faint grid plus fine random texture everywhere,
        # and a strong grid in the top quarter. Local calibration should keep the
        # faint region near the floor and only raise the top tiles.
        h, w = 512, 512
        base = stubs.smooth_image(h, w, seed=11)
        # sparse bright specks: real high-amplitude 2px-band detail, the kind a
        # low clamp protects and a high clamp shaves
        g = torch.Generator().manual_seed(5)
        texture = (torch.rand(1, 1, h, w, generator=g) < 0.01).float().expand(1, 3, h, w) * 0.2
        img = stubs.add_lattice((base + texture).clamp(0, 1), 0.6, 0.6)
        strong = stubs.add_lattice((base + texture).clamp(0, 1), 3.0, 3.0)  # within reach of the 0.05 ceiling even on a speck
        img[:, :, : h // 4, :] = strong[:, :, : h // 4, :]
        x = img.permute(0, 2, 3, 1)
        corr = core.extract_grid(img)
        lim_map, tiles = core.auto_limit_map(corr, tile=128)
        self.assertEqual(tuple(lim_map.shape), (1, 1, h, w))
        self.assertEqual(tuple(tiles.shape), (1, 4, 4))
        self.assertGreater(tiles[0, 0].min().item(), tiles[0, 1:].max().item() * 1.5)  # top row raised, rest not
        self.assertLess(tiles[0, 1:].max().item(), 0.03)
        cleaned, _, st = core.degrid(x, mode="auto", tile=128)
        self.assertLess(st[0]["limit_min"], st[0]["limit_max"])
        self.assertLess(st[0]["residual_255"], 0.25 + 1e-6)
        self.assertLess(lattice_255(cleaned[:, h // 4 :]), 0.3)  # faint region clean too
        self.assertIn("-", core.status_line("auto", st).split("(limit ")[1].split(" ")[0])  # range shown
        # at the specks in the faint region, the local cap shaves far less than one
        # frame-wide ceiling limit would (the faint grid itself is removed by both)
        wide, _, _ = core.degrid(x, mode="manual", limit=0.05)
        specks = texture[0, 0, h // 4 :] > 0
        loss_local = (cleaned - x)[0, h // 4 :][specks].abs().mean().item()
        loss_wide = (wide - x)[0, h // 4 :][specks].abs().mean().item()
        self.assertLess(loss_local, loss_wide * 0.5)
        # small images fall back to one global limit; tiny ones do not crash
        _, _, st_small = core.degrid(x[:, :96, :96], mode="auto")
        self.assertAlmostEqual(st_small[0]["limit_min"], st_small[0]["limit_max"], places=7)
        core.degrid(x[:, :6, :6], mode="auto")
        # batch of two: maps are independent
        both = torch.cat([x, stubs.add_lattice(base).permute(0, 2, 3, 1)], dim=0)
        _, _, st_both = core.degrid(both, mode="auto", tile=128)
        self.assertAlmostEqual(st_both[0]["limit_max"], st[0]["limit_max"], places=6)
        self.assertLess(st_both[1]["limit_max"], st[0]["limit_max"])

    def test_residual_reports_what_the_clamp_left(self):
        x = stubs.add_lattice(stubs.smooth_image(H, W)).permute(0, 2, 3, 1)
        cleaned, _, st = core.degrid(x, mode="manual", limit=0.001)
        self.assertGreater(st[0]["residual_255"], 1.0)
        self.assertAlmostEqual(st[0]["residual_255"], lattice_255(cleaned), delta=0.05)
        self.assertIn("partially removed", core.status_line("manual", st))
        cleaned, _, st = core.degrid(x, mode="manual", limit=0.05)
        self.assertLess(st[0]["residual_255"], 0.1)
        self.assertAlmostEqual(st[0]["residual_255"], lattice_255(cleaned), delta=0.05)
        self.assertIn("— removed", core.status_line("manual", st))
        _, _, st = core.degrid(stubs.smooth_image(H, W, seed=3).permute(0, 2, 3, 1))
        self.assertTrue(st[0]["skipped"])
        self.assertAlmostEqual(st[0]["residual_255"], st[0]["amp_255"], places=6)


class ConfigTests(unittest.TestCase):
    def test_infotext_roundtrip(self):
        cfg = vw.DeGridConfig(mode="manual", limit=0.03, skip_when_clean=False, threshold_255=0.8)
        text = cfg.infotext()
        self.assertEqual(text, "mode=manual;limit=0.03;skip=0;threshold=0.8")
        self.assertNotIn(",", text)
        self.assertNotIn(":", text)
        self.assertEqual(vw.DeGridConfig.parse_infotext(text), {"mode": "manual", "limit": "0.03", "skip": "0", "threshold": "0.8"})
        self.assertEqual(vw.DeGridConfig().infotext(), "mode=auto;skip=1;threshold=0.5")
        self.assertIsNone(vw.DeGridConfig.parse_infotext(None))
        self.assertIsNone(vw.DeGridConfig.parse_infotext("garbage"))


class WrapperTests(unittest.TestCase):
    def make(self, image5d, config=None, **vae_kwargs):
        records = []
        source = stubs.FakeVAE(stubs.FakeDecoder(image5d), **vae_kwargs)
        wrapped = vw.wrap(source, config or vw.DeGridConfig(), core, records.append)
        return source, wrapped, records

    def test_wrap_shares_state_and_keeps_type(self):
        source, wrapped, _ = self.make(gridded_5d())
        self.assertIsInstance(wrapped, stubs.FakeVAE)
        self.assertTrue(vw.is_wrapped(wrapped))
        self.assertFalse(vw.is_wrapped(source))
        self.assertIs(wrapped.first_stage_model, source.first_stage_model)
        self.assertIs(wrapped.patcher, source.patcher)
        self.assertEqual(type(wrapped).__name__, "DeGridFakeVAE")
        # the same class is reused, and wrapping a wrapper unwraps first
        again = vw.wrap(wrapped, vw.DeGridConfig(), core, lambda r: None)
        self.assertIs(type(again), type(wrapped))
        self.assertIs(again.degrid_source, source)
        # encode is untouched: the source class's own method, on shared model state
        self.assertIs(type(wrapped).encode, stubs.FakeVAE.encode)
        wrapped.encode(torch.zeros(1, 8, 8, 3))
        self.assertEqual(wrapped.encode_calls, 1)

    def test_restore_undoes_lora_leak(self):
        source, wrapped, _ = self.make(gridded_5d())
        model = stubs.FakeModel(source)
        model.forge_objects.vae = wrapped
        model.forge_objects_after_applying_lora.vae = wrapped  # what a LoRA re-apply bakes in
        self.assertEqual(vw.restore(model), 2)
        self.assertIs(model.forge_objects.vae, source)
        self.assertIs(model.forge_objects_after_applying_lora.vae, source)
        self.assertEqual(vw.restore(model), 0)
        self.assertEqual(vw.restore(None), 0)

    def test_decode_removes_lattice_and_matches_core(self):
        source, wrapped, records = self.make(gridded_5d())
        latent = torch.zeros(2, 16, 1, H // 8, W // 8)
        base = source.decode(latent)  # [B, T, H, W, C]
        out = wrapped.decode(latent)
        self.assertEqual(out.shape, base.shape)
        self.assertEqual(out.dtype, base.dtype)
        self.assertGreater(lattice_255(base[:, 0]), 2.0)
        self.assertLess(lattice_255(out[:, 0]), 0.3)
        # exactly what the node would do to the same pixels
        expected, _, _ = core.degrid(base.reshape(-1, H, W, 3))
        self.assertTrue(torch.allclose(out.reshape(-1, H, W, 3), expected, atol=1e-6))
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual((rec.height, rec.width, rec.count), (H, W, 2))
        self.assertFalse(rec.stats[0]["skipped"])
        self.assertEqual(rec.previews, [None, None])

    def test_clean_image_passes_through_bit_for_bit(self):
        source, wrapped, records = self.make(clean_5d())
        latent = torch.zeros(1, 16, 1, H // 8, W // 8)
        base = source.decode(latent)
        out = wrapped.decode(latent)
        self.assertTrue(torch.equal(out, base))
        self.assertTrue(records[0].stats[0]["skipped"])

    def test_four_d_non_wan_and_manual_limit(self):
        image4d = stubs.add_lattice(stubs.smooth_image(H, W))
        cfg = vw.DeGridConfig(mode="manual", limit=0.03)
        source, wrapped, records = self.make(image4d, cfg, is_wan=False)
        latent = torch.zeros(1, 4, H // 8, W // 8)
        out = wrapped.decode(latent)
        self.assertEqual(tuple(out.shape), (1, H, W, 3))
        self.assertLess(lattice_255(out), 0.3)
        self.assertAlmostEqual(records[0].stats[0]["limit"], 0.03, places=6)

    def test_threshold_and_skip_flag(self):
        cfg = vw.DeGridConfig(threshold_255=10.0)  # everything counts as clean
        source, wrapped, records = self.make(gridded_5d(), cfg)
        latent = torch.zeros(1, 16, 1, H // 8, W // 8)
        self.assertTrue(torch.equal(wrapped.decode(latent), source.decode(latent)))
        self.assertTrue(records[0].stats[0]["skipped"])
        cfg = vw.DeGridConfig(threshold_255=10.0, skip_when_clean=False)  # ...but filter anyway
        source, wrapped, records = self.make(gridded_5d(), cfg)
        self.assertFalse(torch.equal(wrapped.decode(latent), source.decode(latent)))
        self.assertFalse(records[0].stats[0]["skipped"])

    def test_tiled_and_oom_fallback_make_one_record(self):
        source, wrapped, records = self.make(gridded_5d())
        latent = torch.zeros(1, 16, 1, H // 8, W // 8)
        wrapped.decode_tiled(latent)
        self.assertEqual(len(records), 1)
        wrapped.oom_once = True  # decode() -> decode_tiled() nested, still one record
        out = wrapped.decode(latent)
        self.assertEqual(len(records), 2)
        self.assertEqual(wrapped.tiled_calls, 2)
        self.assertLess(lattice_255(out[:, 0]), 0.3)

    def test_previews_are_zoomed_centre_crops(self):
        cfg = vw.DeGridConfig(preview=True, preview_zoom=4, grid_gain=10.0)
        source, wrapped, records = self.make(gridded_5d(), cfg)
        wrapped.decode(torch.zeros(1, 16, 1, H // 8, W // 8))
        vis = records[0].previews[0]
        self.assertEqual(tuple(vis.shape), (H // 4 * 4, W // 4 * 4, 3))
        self.assertEqual(vis.device.type, "cpu")
        # the removed stripes show up as alternating columns around mid-grey;
        # after 4x nearest magnification each source column is a 4-wide block
        self.assertGreater((vis[:, 0::8] - vis[:, 4::8]).abs().mean().item(), 0.05)


class ScriptTests(unittest.TestCase):
    UI_ORDER = ["enabled", "mode", "limit", "skip_when_clean", "threshold", "preview", "preview_zoom", "grid_gain"]
    DEFAULTS = dict(enabled=False, mode="auto", limit=0.02, skip_when_clean=True, threshold=0.5, preview=False, preview_zoom="8x", grid_gain=10.0)

    def setUp(self):
        self.script = script_mod.DeGridScript()
        script_mod.DeGridScript._records.clear()
        script_mod.DeGridScript._previews.clear()
        script_mod.DeGridScript._config = None

    def args(self, **overrides):
        values = dict(self.DEFAULTS, **overrides)
        return [values[name] for name in self.UI_ORDER]

    def test_ui_order_matches_hook_signature_and_defaults(self):
        controls = self.script.ui(False)
        params = [n for n in inspect.signature(self.script.process_before_every_sampling).parameters if n not in ("p", "kwargs")]
        self.assertEqual(params, self.UI_ORDER)
        self.assertEqual(len(controls), len(self.UI_ORDER))
        for control, name in zip(controls, self.UI_ORDER):
            self.assertEqual(control.value, self.DEFAULTS[name], name)
        self.assertEqual(len(self.script.infotext_fields), 5)

    def test_disabled_is_a_no_op(self):
        source = stubs.FakeVAE(stubs.FakeDecoder(gridded_5d()))
        p = stubs.FakeProcessing(source)
        self.script.process_before_every_sampling(p, *self.args(enabled=False))
        self.assertIs(p.sd_model.forge_objects.vae, source)
        self.assertNotIn(script_mod.INFOTEXT_KEY, p.extra_generation_params)
        self.script.postprocess_batch(p)
        self.assertNotIn(script_mod.INFOTEXT_RESULT_KEY, p.extra_generation_params)

    def test_full_run_with_hires_style_double_pass(self):
        source = stubs.FakeVAE(stubs.FakeDecoder(gridded_5d()))
        p = stubs.FakeProcessing(source)
        latent = torch.zeros(1, 16, 1, H // 8, W // 8)

        self.script.process_before_every_sampling(p, *self.args(enabled=True, preview=True, preview_zoom="4x"))
        self.assertTrue(vw.is_wrapped(p.sd_model.forge_objects.vae))
        self.assertEqual(p.extra_generation_params[script_mod.INFOTEXT_KEY], "mode=auto;skip=1;threshold=0.5")
        first_pass = p.sd_model.forge_objects.vae.decode(latent)  # hires first-pass decode

        # Forge resets forge_objects before the second pass; the hook re-wraps
        p.sd_model.forge_objects = p.sd_model.forge_objects_after_applying_lora.shallow_copy()
        self.script.process_before_every_sampling(p, *self.args(enabled=True, preview=True, preview_zoom="4x"))
        self.assertTrue(vw.is_wrapped(p.sd_model.forge_objects.vae))
        final = p.sd_model.forge_objects.vae.decode(latent)
        self.assertLess(lattice_255(first_pass[:, 0]), 0.3)
        self.assertLess(lattice_255(final[:, 0]), 0.3)

        with self.assertLogs("processing", level="INFO") as logs:
            self.script.postprocess_batch(p)
        self.assertEqual(len(logs.output), 2, logs.output)  # one line per decode
        self.assertIn(f"DeGrid: {W}x{H}", logs.output[0])
        self.assertTrue(all(line.isascii() for line in logs.output), logs.output)
        result = p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY]
        self.assertTrue(result.startswith("grid "))
        self.assertIn("removed", result)
        self.assertTrue(result.isascii())
        self.assertEqual(len(script_mod.DeGridScript._previews), 1)  # from the final decode only

        processed = stubs.FakeProcessed()
        self.script.postprocess(p, processed)
        self.assertIs(p.sd_model.forge_objects.vae, source)
        self.assertEqual(len(processed.extra_images), 1)
        self.assertEqual(processed.extra_images[0].size, (W // 4 * 4, H // 4 * 4))
        self.assertEqual(len(processed.infotexts), 2)
        self.assertEqual(processed.infotexts[1], processed.infotexts[0])
        self.assertEqual(script_mod.DeGridScript._previews, [])
        self.assertIsNone(script_mod.DeGridScript._config)

    def test_restore_runs_before_rewrap_and_on_lora_leak(self):
        source = stubs.FakeVAE(stubs.FakeDecoder(gridded_5d()))
        p = stubs.FakeProcessing(source)
        self.script.process_before_every_sampling(p, *self.args(enabled=True))
        leaked = p.sd_model.forge_objects.vae
        p.sd_model.forge_objects_after_applying_lora.vae = leaked
        # next run, extension disabled: the leak must still be cleaned up
        p.sd_model.forge_objects = p.sd_model.forge_objects_after_applying_lora.shallow_copy()
        self.script.process_before_every_sampling(p, *self.args(enabled=False))
        self.assertIs(p.sd_model.forge_objects.vae, source)
        self.assertIs(p.sd_model.forge_objects_after_applying_lora.vae, source)

    def test_paste_fields(self):
        self.script.ui(False)
        readers = {name: pf.function for pf, name in zip(self.script.infotext_fields, ["enabled", "mode", "limit", "skip", "threshold"])}
        params = {"DeGrid": "mode=manual;limit=0.03;skip=0;threshold=0.8"}
        self.assertIs(readers["enabled"](params), True)
        self.assertEqual(readers["mode"](params), "manual")
        self.assertAlmostEqual(readers["limit"](params), 0.03)
        self.assertIs(readers["skip"](params), False)
        self.assertAlmostEqual(readers["threshold"](params), 0.8)
        self.assertIs(readers["enabled"]({}), False)
        self.assertIsNone(readers["mode"]({}))
        self.assertIsNone(readers["limit"]({"DeGrid": "mode=auto;skip=1;threshold=0.5"}))

    def test_build_config_tolerates_odd_inputs(self):
        cfg = script_mod.DeGridScript.build_config("bogus", "0.05", 0, "1.2", 1, "weird", 20)
        self.assertEqual(cfg.mode, "auto")
        self.assertEqual(cfg.limit, 0.05)
        self.assertFalse(cfg.skip_when_clean)
        self.assertAlmostEqual(cfg.threshold, 1.2 / 255.0)
        self.assertEqual(cfg.preview_zoom, 8)
        self.assertEqual(cfg.grid_gain, 20.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
