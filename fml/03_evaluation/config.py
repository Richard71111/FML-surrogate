"""Config shim — imports Config from case root (../../config.py)."""
import os, importlib.util as _ilu
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
_spec = _ilu.spec_from_file_location("_root_config", os.path.join(_root, "config.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Config = _mod.Config
