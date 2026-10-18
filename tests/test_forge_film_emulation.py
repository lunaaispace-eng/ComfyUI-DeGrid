"""Offline tests for the Forge Neo film emulation script (hook wiring, parameters, seed, paste round trip)."""

from __future__ import annotations

import inspect
import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_stubs as stubs  # noqa: E402

stubs.install_fakes()

from lib_degrid.loader import load_film_core  # noqa: E402

fg = load_film_core(str(stubs.REPO_ROOT))
script_mod = stubs.load_script_module(stubs.REPO_ROOT / "scripts" / "film_emulation_forge.py", "film_emulation_forge_under_test")

H, W = 96, 128


def pil_image(value: float = 0.4, rgb=(1.0, 1.0, 1.0)) -> Image.Image:
    arr = np.full((H, W, 3), value, np.float32) * np.array(rgb, np.float32)
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGB")


def as_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]


class ScriptTests(unittest.TestCase):
    UI_ORDER = ["enabled", "iso", "grain", "colour_grain", "match_texture", "film_type", "softness", "halation", "bloom", "highlight_rolloff"]
    DEFAULTS = dict(enabled=False, iso="ISO 400", grain=50, colour_grain="auto", match_texture=True, film_type="print",
                    softness=0, halation=0, bloom=0, highlight_rolloff=0)

    def setUp(self):
        self.script = script_mod.FilmEmulationScript()
        self.pp_cls = stubs.fake_forge_modules(str(stubs.REPO_ROOT))["modules.scripts"].PostprocessImageArgs

    def args(self, **overrides):
        values = dict(self.DEFAULTS, **overrides)
        return [values[name] for name in self.UI_ORDER]

    def processing(self, seeds=(1234, 1235)):
        p = stubs.FakeProcessing(stubs.FakeVAE(stubs.FakeDecoder(torch.zeros(1, 3, 1, H, W))))
        p.all_seeds = list(seeds)
        p.seed = seeds[0]
        return p

    def test_ui_order_matches_hooks_and_defaults(self):
        controls = self.script.ui(False)
        for hook in (self.script.process, self.script.postprocess_image_after_composite):
            params = [n for n in inspect.signature(hook).parameters if n not in ("p", "pp", "kwargs")]
            self.assertEqual(params, self.UI_ORDER, hook.__name__)
        self.assertEqual(len(controls), len(self.UI_ORDER))
        for control, name in zip(controls, self.UI_ORDER):
            self.assertEqual(control.value, self.DEFAULTS[name], name)
        self.assertEqual(controls[1].choices, list(fg.ISO_NAMES))
        self.assertEqual(controls[3].choices, list(fg.CHROMA_MODES))
        self.assertEqual(self.script.sorting_priority, 30)
        self.assertEqual(len(self.script.infotext_fields), 10)
        self.assertEqual(controls[5].choices, list(fg.FILM_TYPES))

    def test_disabled_is_a_no_op(self):
        p = self.processing()
        image = pil_image()
        pp = self.pp_cls(image, 0)
        self.script.process(p, *self.args())
        self.script.postprocess_image_after_composite(p, pp, *self.args())
        self.assertIs(pp.image, image)
        self.assertNotIn(script_mod.INFOTEXT_KEY, p.extra_generation_params)

    def test_full_run_uses_the_image_seed_and_writes_parameters(self):
        p = self.processing(seeds=(7, 8))
        image = pil_image(fg.MIDTONE_REF)
        pp = self.pp_cls(image, 1)  # second image of the batch
        self.script.process(p, *self.args(enabled=True, match_texture=False))
        self.assertEqual(p.extra_generation_params[script_mod.INFOTEXT_KEY], "iso=ISO400;grain=50;chroma=auto;match=0;film=print;soft=0;halation=0;bloom=0;rolloff=0")
        with self.assertLogs("processing", level="INFO") as logs:
            self.script.postprocess_image_after_composite(p, pp, *self.args(enabled=True, match_texture=False))
        self.assertEqual(len(logs.output), 1)
        self.assertIn(f"Film emulation: {W}x{H} [2]", logs.output[0])
        self.assertIn("seed 8", logs.output[0])
        self.assertTrue(logs.output[0].isascii(), logs.output[0])
        result = p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY]
        self.assertTrue(result.startswith("grain ISO 400 strength 50"))
        self.assertNotIn(":", result)
        self.assertIn("monochrome (luma grain only)", result)  # a grey card
        self.assertNotIn(",", result)  # Forge would JSON-quote the value

        self.assertIsNot(pp.image, image)
        x_in, x_out = as_tensor(image), as_tensor(pp.image)
        d = (x_out - x_in)[..., 0] * 255.0
        self.assertAlmostEqual(d.std().item(), fg.strength_to_amp(50), delta=0.5)  # midtone grey at the bell's peak
        self.assertTrue(torch.equal(x_out[..., 0], x_out[..., 1]))  # grey stays grey
        # exactly what the core would produce with that seed
        expected, _ = fg.add_grain(x_in, "ISO 400", 50, seed=8, match_texture=False)
        self.assertTrue(torch.allclose(x_out, expected, atol=1.0 / 255))

    def test_match_texture_gives_a_clean_image_the_full_amount(self):
        p = self.processing(seeds=(7,))
        image = pil_image(fg.MIDTONE_REF)  # a perfectly clean card: floor 0
        pp = self.pp_cls(image, 0)
        self.script.process(p, *self.args(enabled=True))
        self.assertIn(";match=1", p.extra_generation_params[script_mod.INFOTEXT_KEY])
        self.script.postprocess_image_after_composite(p, pp, *self.args(enabled=True))
        result = p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY]
        self.assertIn("x1.00 for texture floor 0.00/255", result)
        d = (as_tensor(pp.image) - as_tensor(image))[..., 0] * 255.0
        self.assertAlmostEqual(d.std().item(), fg.strength_to_amp(50), delta=0.5)

    def test_seed_fallbacks(self):
        p = self.processing(seeds=(5,))
        self.assertEqual(script_mod.seed_for(p, 0), 5)
        self.assertEqual(script_mod.seed_for(p, 3), 5 + 3)  # beyond all_seeds: base seed plus index

        class Bare:
            pass

        self.assertEqual(script_mod.seed_for(Bare(), 2), 2)

    def test_tolerates_odd_inputs_and_rgba(self):
        p = self.processing()
        image = pil_image(0.5, rgb=(1.0, 0.8, 0.6)).convert("RGBA")
        pp = self.pp_cls(image, 0)
        self.script.postprocess_image_after_composite(p, pp, True, "ISO 12800", "75", "maybe", True, "slide", "0", 0, 0, 0)
        self.assertEqual(pp.image.mode, "RGBA")
        self.assertTrue(np.array_equal(np.asarray(pp.image)[..., 3], np.asarray(image)[..., 3]))
        self.assertIn("grain ISO 400 strength 75", p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY])
        self.assertIn("colour grain 8%", p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY])

    def test_optics_run_and_are_written(self):
        p = self.processing(seeds=(7,))
        image = pil_image(0.5)
        image.paste((255, 255, 255), (50, 30, 80, 60))  # a light source
        pp = self.pp_cls(image, 0)
        args = self.args(enabled=True, grain=0, softness=30, halation=40, bloom=20, highlight_rolloff=50)
        self.script.process(p, *args)
        self.assertTrue(p.extra_generation_params[script_mod.INFOTEXT_KEY].endswith(";soft=30;halation=40;bloom=20;rolloff=50"))
        self.script.postprocess_image_after_composite(p, pp, *args)
        result = p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY]
        self.assertTrue(result.startswith("grain off · softness 30 halation 40 bloom 20 roll-off 50".replace("·", "|")), result)
        expected, _ = fg.emulate(as_tensor(image), grain=0, seed=7, softness=30, halation=40, bloom=20, rolloff=50)
        self.assertTrue(torch.allclose(as_tensor(pp.image), expected, atol=1.0 / 255))

    def test_paste_fields(self):
        self.script.ui(False)
        readers = {name: pf.function for pf, name in zip(self.script.infotext_fields, ["enabled", "iso", "grain", "chroma", "match", "film", "soft", "halation", "bloom", "rolloff"])}
        params = {script_mod.INFOTEXT_KEY: "iso=ISO3200;grain=35;chroma=off;match=0;film=negative"}
        self.assertIs(readers["enabled"](params), True)
        self.assertEqual(readers["iso"](params), "ISO 3200")
        self.assertAlmostEqual(readers["grain"](params), 35.0)
        self.assertEqual(readers["chroma"](params), "off")
        self.assertIs(readers["match"](params), False)
        self.assertEqual(readers["film"](params), "negative")
        self.assertIsNone(readers["film"]({script_mod.INFOTEXT_KEY: "iso=ISO400;grain=50;chroma=auto;match=1"}))  # older images
        # images made before the rename: "Film grain: ...;strength=..." restores into this accordion
        legacy = {script_mod.LEGACY_INFOTEXT_KEY: "iso=ISO1600;strength=65;chroma=on;match=1"}
        self.assertIs(readers["enabled"](legacy), True)
        self.assertEqual(readers["iso"](legacy), "ISO 1600")
        self.assertAlmostEqual(readers["grain"](legacy), 65.0)
        self.assertEqual(readers["chroma"](legacy), "on")
        self.assertIs(readers["match"](legacy), True)
        self.assertIsNone(readers["film"](legacy))  # the old stage had no film type: keep the default
        self.assertIsNone(readers["match"]({script_mod.LEGACY_INFOTEXT_KEY: "iso=ISO400;strength=50;chroma=auto"}))  # before match_texture
        both = dict(legacy, **params)  # the new key wins
        self.assertAlmostEqual(readers["grain"](both), 35.0)
        # optics: read when written, 0 when the infotext predates them, untouched when the stage was off
        full = {script_mod.INFOTEXT_KEY: "iso=ISO400;grain=50;chroma=auto;match=1;film=print;soft=30;halation=40;bloom=20;rolloff=50"}
        self.assertEqual([readers[k](full) for k in ("soft", "halation", "bloom", "rolloff")], [30.0, 40.0, 20.0, 50.0])
        self.assertEqual([readers[k](params) for k in ("soft", "halation", "bloom", "rolloff")], [0.0] * 4)
        self.assertEqual([readers[k](legacy) for k in ("soft", "halation", "bloom", "rolloff")], [0.0] * 4)
        self.assertEqual([readers[k]({}) for k in ("soft", "halation", "bloom", "rolloff")], [None] * 4)
        self.assertIs(readers["enabled"]({}), False)
        self.assertIsNone(readers["iso"]({}))
        self.assertIsNone(script_mod.parse_infotext("garbage"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
