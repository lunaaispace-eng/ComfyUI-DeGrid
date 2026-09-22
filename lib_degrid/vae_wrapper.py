"""Decode-time DeGrid for Forge Neo: a VAE wrapper that notch-filters every decode.

Why the VAE and not an image hook: Forge's hires fix decodes the first pass and
hands the pixels straight to the upscaler with no script hook in between
(``modules/processing.py::sample_hr_pass``). An image-level hook would leave the
2px lattice in exactly the place the node's README warns about — under an
upscaler that reads it as detail. Every decode path in ``backend/patcher/vae.py``
(direct, tiled, the out-of-memory fallback to tiled) funnels through
``process_output``, so overriding that one method covers them all.

The wrapper is a dynamically created subclass of the *source VAE's own class*
(``DeGrid<SourceClass>``), built with the source's ``__dict__`` copied over. It
therefore shares the source's weights and patcher (no second model load), keeps
``isinstance`` checks against the source class true, and composes with other
extensions that subclass ``VAE`` (their ``process_output`` runs first via
``super()``). No ``backend`` import is needed here, which also keeps this module
testable without a Forge checkout.

Usage from a ``scripts.Script``::

    restore(p.sd_model)                       # undo any earlier swap first
    vae = wrap(p.sd_model.forge_objects.vae, config, core, records.append)
    p.sd_model.forge_objects.vae = vae        # scoped to the next sampling pass
    ...
    restore(p.sd_model)                       # in postprocess()

Forge re-copies ``forge_objects`` from ``forge_objects_after_applying_lora``
before every ``process_before_every_sampling`` call, so the swap normally dies
on its own — except that applying a LoRA rebuilds that copy from the *live*
objects and only resets ``unet``/``clip`` (``sd_forge_lora/networks.py``). Hence
``restore()`` checks both bundles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

NEGLIGIBLE_AMP_255 = 0.5  # mirrors degrid_core.NEGLIGIBLE_AMP, in /255 units


@dataclass
class DeGridConfig:
    mode: str = "auto"
    limit: float = 0.02  # manual mode only, 0..1 units
    skip_when_clean: bool = True
    threshold_255: float = NEGLIGIBLE_AMP_255  # "clean" below this lattice amplitude
    preview: bool = False
    preview_zoom: int = 8
    grid_gain: float = 10.0

    @property
    def threshold(self) -> float:
        return float(self.threshold_255) / 255.0

    def infotext(self) -> str:
        """Compact ``k=v;k=v`` form for the PNG parameters. No commas or colons,
        so Forge writes it unquoted."""
        parts = [f"mode={self.mode}"]
        if self.mode == "manual":
            parts.append(f"limit={float(self.limit):g}")
        parts.append(f"skip={int(bool(self.skip_when_clean))}")
        parts.append(f"threshold={float(self.threshold_255):g}")
        return ";".join(parts)

    @staticmethod
    def parse_infotext(text: str | None) -> dict[str, str] | None:
        """Inverse of ``infotext()``: raw string values keyed by name, or None."""
        if not text or not isinstance(text, str):
            return None
        out: dict[str, str] = {}
        for part in text.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out or None


@dataclass
class DecodeRecord:
    """Everything one decode call produced, for logging / infotext / previews."""

    height: int
    width: int
    stats: list[dict[str, Any]] = field(default_factory=list)  # one per image
    previews: list[torch.Tensor | None] = field(default_factory=list)  # [h, w, c] CPU float, or None

    @property
    def count(self) -> int:
        return len(self.stats)


_OOM = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)


class DeGridMixin:
    """Mixed in front of the source VAE class; see module docstring."""

    degrid_source: Any
    degrid_config: DeGridConfig
    degrid_core: Any
    degrid_sink: Callable[[DecodeRecord], None]
    _degrid_depth: int
    _degrid_pending: list

    # -- grouping: one DecodeRecord per public decode call -------------------

    def decode(self, samples_in, *args, **kwargs):
        return self._degrid_grouped(super().decode, samples_in, *args, **kwargs)

    def decode_tiled(self, samples, *args, **kwargs):
        return self._degrid_grouped(super().decode_tiled, samples, *args, **kwargs)

    def _degrid_grouped(self, fn, *args, **kwargs):
        # decode() falls back to decode_tiled() on OOM, and the direct path
        # calls process_output once per sub-batch: a depth counter turns all
        # of that into a single record per outermost call.
        self._degrid_depth += 1
        try:
            return fn(*args, **kwargs)
        finally:
            self._degrid_depth -= 1
            if self._degrid_depth == 0 and self._degrid_pending:
                pending, self._degrid_pending = self._degrid_pending, []
                h, w = pending[0][2]
                record = DecodeRecord(height=h, width=w)
                for stats, preview, _ in pending:
                    record.stats.append(stats)
                    record.previews.append(preview)
                self.degrid_sink(record)

    # -- the funnel ----------------------------------------------------------

    def process_output(self, image):
        image = super().process_output(image)  # source class first (0..1 range, any pixel shuffle, ...)
        return self._degrid_apply(image)

    def _degrid_device(self, x: torch.Tensor) -> torch.device:
        if x.device.type != "cpu":
            return x.device
        dev = getattr(self, "device", None)
        if isinstance(dev, torch.device) and dev.type != "cpu":
            return dev
        return x.device

    def _degrid_apply(self, image: torch.Tensor) -> torch.Tensor:
        """image: [B, C, H, W] or Wan-style [B, C, T, H, W], float, 0..1."""
        if not isinstance(image, torch.Tensor) or image.ndim not in (4, 5):
            return image

        five_d = image.ndim == 5
        if five_d:
            b, c, t, h, w = image.shape
            x = image.movedim(1, -1).reshape(b * t, h, w, c)
        else:
            b, c, h, w = image.shape
            x = image.movedim(1, -1)

        cfg = self.degrid_config
        core = self.degrid_core
        out = torch.empty_like(x)
        for i in range(x.shape[0]):
            xi = x[i : i + 1]
            dev = self._degrid_device(xi)
            try:
                cleaned, vis, stats = self._degrid_one(xi.to(dev), cfg, core)
            except _OOM:
                if dev.type == "cpu":
                    raise
                torch.cuda.empty_cache()
                cleaned, vis, stats = self._degrid_one(xi.to("cpu"), cfg, core)
            out[i : i + 1] = cleaned.to(x.device)
            preview = None
            if cfg.preview:
                zoom = int(cfg.preview_zoom)
                preview = (core.zoom_center(vis, zoom) if zoom > 1 else vis)[0].to("cpu")
            self._degrid_pending.append((stats[0], preview, (h, w)))

        if five_d:
            return out.reshape(b, t, h, w, c).movedim(-1, 1)
        return out.movedim(-1, 1)

    @staticmethod
    def _degrid_one(xi: torch.Tensor, cfg: DeGridConfig, core):
        return core.degrid(
            xi,
            mode=cfg.mode,
            limit=float(cfg.limit),
            grid_gain=float(cfg.grid_gain),
            grid_view="full frame",
            skip_when_clean=bool(cfg.skip_when_clean),
            threshold=cfg.threshold,
        )


_wrapper_classes: dict[type, type] = {}


def is_wrapped(obj) -> bool:
    return isinstance(obj, DeGridMixin)


def unwrap(obj):
    while isinstance(obj, DeGridMixin):
        obj = obj.degrid_source
    return obj


def wrap(source, config: DeGridConfig, core, sink: Callable[[DecodeRecord], None]):
    """Return a DeGrid-ing stand-in for ``source`` that shares all its state."""
    source = unwrap(source)
    src_cls = type(source)
    cls = _wrapper_classes.get(src_cls)
    if cls is None:
        cls = type(f"DeGrid{src_cls.__name__}", (DeGridMixin, src_cls), {"__module__": __name__})
        _wrapper_classes[src_cls] = cls

    obj = cls.__new__(cls)  # skip VAE.__init__: no model to build, we borrow the source's
    obj.__dict__.update(source.__dict__)
    obj.degrid_source = source
    obj.degrid_config = config
    obj.degrid_core = core
    obj.degrid_sink = sink
    obj._degrid_depth = 0
    obj._degrid_pending = []
    return obj


def restore(sd_model) -> int:
    """Put the source VAE back wherever a wrapper ended up. Returns how many were undone."""
    if sd_model is None:
        return 0
    original = getattr(sd_model, "forge_objects_original", None)
    undone = 0
    for name in ("forge_objects", "forge_objects_after_applying_lora"):
        objects = getattr(sd_model, name, None)
        vae = getattr(objects, "vae", None) if objects is not None else None
        if is_wrapped(vae):
            objects.vae = unwrap(vae) or (original.vae if original is not None else None)
            undone += 1
    return undone
