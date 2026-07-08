"""Lightweight YAML config with env + interpolation + CLI overrides.

Supports:
  * ``${env:VAR:default}``            -> os.environ.get("VAR", "default")
  * ``${paths.data_root}``            -> reference another resolved key
  * ``--set a.b.c=value`` overrides   -> nested assignment with type inference
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List

import yaml

_ENV_RE = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
_REF_RE = re.compile(r"\$\{([a-zA-Z_][\w.]*)\}")


class Config(dict):
    """dict with attribute access and dotted get/set."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_dotted(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_dotted(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: Dict[str, Any] = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _infer(value: str) -> Any:
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _resolve(obj: Any, root: Dict) -> Any:
    """Resolve env + reference placeholders. Run repeatedly until stable."""
    if isinstance(obj, dict):
        return {k: _resolve(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve(v, root) for v in obj]
    if isinstance(obj, str):
        s = _ENV_RE.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), obj)

        def ref(m: "re.Match") -> str:
            node: Any = root
            for part in m.group(1).split("."):
                node = node[part] if isinstance(node, dict) and part in node else ""
            return str(node)

        return _REF_RE.sub(ref, s)
    return obj


def load_config(
    config_path: str,
    dataset_config: str | None = None,
    overrides: List[str] | None = None,
) -> Config:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    if dataset_config and os.path.exists(dataset_config):
        with open(dataset_config, "r", encoding="utf-8") as f:
            _deep_merge(cfg, yaml.safe_load(f) or {})

    cfg = Config(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, val = item.split("=", 1)
        cfg.set_dotted(key.strip(), _infer(val.strip()))

    # resolve twice so ${paths.x} that itself uses ${env:...} settles
    resolved = _resolve(cfg, cfg)
    resolved = _resolve(resolved, resolved)
    return Config(resolved)


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to base YAML config")
    parser.add_argument("--dataset-config", default=None, help="optional per-dataset YAML")
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides a.b=c")
