"""Offline check of the ComfyUI node definitions, without ComfyUI.

A minimal fake of ``comfy_api.latest`` (just enough of ``io`` / ``ui`` to build
the schemas) lets the repo root import as a package. The check that matters is
the one ComfyUI itself never makes: every input id declared in the schema must
match ``execute()``'s parameters, in order, or the node silently gets its
widgets shuffled.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
import forge_stubs as stubs  # noqa: E402


def install_fake_comfy_api():
    class _Input:
        def __init__(self, id, **kwargs):
            self.id = id
            self.kwargs = kwargs

    class _Output:
        def __init__(self, id=None, **kwargs):
            self.id = id
            self.kwargs = kwargs

    class _Type:
        Input = _Input
        Output = _Output

    class Schema:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ComfyNode:
        pass

    class NodeOutput:
        def __init__(self, *args, ui=None):
            self.args = args
            self.ui = ui

    io = types.ModuleType("comfy_api.latest.io")
    for name in ("Image", "Vae", "Boolean", "Combo", "Float", "Int", "String"):
        setattr(io, name, type(name, (_Type,), {}))
    io.Schema = Schema
    io.ComfyNode = ComfyNode
    io.NodeOutput = NodeOutput

    ui = types.ModuleType("comfy_api.latest.ui")
    ui.PreviewText = lambda text: {"text": text}

    latest = types.ModuleType("comfy_api.latest")
    latest.io = io
    latest.ui = ui
    latest.ComfyExtension = type("ComfyExtension", (), {})
    pkg = types.ModuleType("comfy_api")
    pkg.latest = latest
    sys.modules["comfy_api"] = pkg
    sys.modules["comfy_api.latest"] = latest
    sys.modules["comfy_api.latest.io"] = io
    sys.modules["comfy_api.latest.ui"] = ui
    try:
        import typing_extensions  # noqa: F401
    except ImportError:
        te = types.ModuleType("typing_extensions")
        te.override = lambda f: f
        sys.modules["typing_extensions"] = te


def load_node_package():
    install_fake_comfy_api()
    name = "comfyui_degrid_under_test"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


node_pkg = load_node_package()


class NodeSchemaTests(unittest.TestCase):
    def check(self, node_cls, expected_id):
        # ComfyUI calls execute(**inputs) keyed by input id, so names must match; order is free.
        schema = node_cls.define_schema()
        self.assertEqual(schema.node_id, expected_id)
        input_ids = [i.id for i in schema.inputs]
        params = [n for n in inspect.signature(node_cls.execute).parameters]
        self.assertEqual(sorted(input_ids), sorted(params), node_cls.__name__)
        self.assertEqual(len(set(input_ids)), len(input_ids))
        return schema

    def test_degrid_schema(self):
        schema = self.check(node_pkg.VAEDeGrid, "VAEDeGrid")
        self.assertEqual([o.id for o in schema.outputs], ["cleaned", "removed_grid"])

    def test_enhance_schema_and_defaults(self):
        schema = self.check(node_pkg.VAEEnhance, "VAEEnhance")
        self.assertEqual([o.id for o in schema.outputs], ["enhanced", "mask"])
        defaults = {i.id: i.kwargs.get("default") for i in schema.inputs}
        core = node_pkg.enhance_core
        self.assertEqual(defaults["gain"], core.GAIN)
        self.assertEqual(defaults["sigma"], core.SIGMA)
        self.assertEqual(defaults["mask_floor"], core.FLOOR)
        self.assertEqual(defaults["texture_target"], core.TARGET)
        self.assertTrue(defaults["skin_only"])
        self.assertTrue(defaults["tone_fix"])
        self.assertTrue(defaults["degrid_first"])
        self.assertFalse(defaults["post_notch"])

    def test_enhance_execute_with_identity_codec(self):
        h, w = 96, 128
        img = stubs.add_lattice(stubs.smooth_image(h, w), 1.5, 1.0).permute(0, 2, 3, 1)
        codec = stubs.FakeFlux2VAE()
        kw = dict(image=img, vae=codec, enabled=True, gain=0.5, sigma=1.0, mask_floor=0.0, texture_target=0.0, skin_only=False, tone_fix=True, degrid_first=True, post_notch=False)
        out = node_pkg.VAEEnhance.execute(**kw)  # keyword call, as ComfyUI does it
        enhanced, mask = out.args
        self.assertEqual(tuple(enhanced.shape), tuple(img.shape))
        self.assertEqual(tuple(mask.shape), tuple(img.shape))
        text = out.ui["text"]
        self.assertIn("removed first", text)  # degrid_first ran on a gridded input
        self.assertIn("output grid", text)
        self.assertNotIn("untested", text)
        corr = node_pkg.extract_grid(enhanced.permute(0, 3, 1, 2).float())
        self.assertLess(node_pkg.lattice_amp(corr)[0].max().item() * 255.0, 0.4)
        # disabled: passthrough
        out = node_pkg.VAEEnhance.execute(**dict(kw, enabled=False))
        self.assertIs(out.args[0], img)
        # an unknown VAE is accepted and flagged
        codec.latent_channels = 4
        out = node_pkg.VAEEnhance.execute(**dict(kw, degrid_first=False))
        self.assertIn("untested VAE", out.ui["text"])
        self.assertNotIn("input grid", out.ui["text"])  # degrid_first off

    def test_extension_lists_both_nodes(self):
        import asyncio

        nodes = asyncio.run(node_pkg.DeGridExtension().get_node_list())
        self.assertEqual([n.__name__ for n in nodes], ["VAEDeGrid", "VAEEnhance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
