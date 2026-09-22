"""Load the repo-root degrid_core.py as a shared module for the Forge Neo script.

degrid_core.py has no ComfyUI imports (torch only), so the exact same file is
the maths behind the ComfyUI node and the Forge Neo extension. It is loaded via
``importlib.util.spec_from_file_location`` under a private ``sys.modules`` key so
it never collides with another extension's module of the same bare filename.

The cache is keyed on the file's mtime and size rather than being permanent:
Forge's "Reload UI" re-executes ``scripts/*.py`` but leaves this module in
``sys.modules``, so a permanent cache would keep serving the core that was on
disk when the process started and a ``git pull`` would appear to do nothing.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_CORE_FILE = "degrid_core.py"
_ENHANCE_FILE = "vae_enhance_core.py"
_MODULE_NAMES = {_CORE_FILE: "comfyui_degrid_core", _ENHANCE_FILE: "comfyui_degrid_enhance_core"}
_cache: dict[str, tuple[tuple, object]] = {}  # file name -> (stamp, module)


def core_stamp(extension_root: str, file_name: str = _CORE_FILE):
    """Identity of a core file currently on disk: (path, mtime_ns, size)."""
    path = os.path.join(extension_root, file_name)
    try:
        info = os.stat(path)
        return (path, info.st_mtime_ns, info.st_size)
    except OSError:
        return (path, None, None)


def _load(extension_root: str, file_name: str):
    stamp = core_stamp(extension_root, file_name)
    cached = _cache.get(file_name)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    name = _MODULE_NAMES[file_name]
    spec = importlib.util.spec_from_file_location(name, stamp[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _cache[file_name] = (stamp, module)
    return module


def load_core(extension_root: str):
    """Import degrid_core.py from ``extension_root``, re-importing it when it changes.

    ``extension_root`` should be captured via ``scripts.basedir()`` at the calling
    script's own import time; ``scripts.basedir()`` reflects whichever script Forge
    is currently loading and is not reliable later, e.g. inside a UI callback.
    """
    return _load(extension_root, _CORE_FILE)


def load_enhance_core(extension_root: str):
    """Same for vae_enhance_core.py (the detail-extrapolation maths)."""
    return _load(extension_root, _ENHANCE_FILE)
