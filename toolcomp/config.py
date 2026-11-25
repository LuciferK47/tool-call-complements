"""YAML config loading with a stable content hash (stored in every result manifest)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_config_hash"] = hashlib.sha256(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                                    sort_keys=True).encode()).hexdigest()[:12]
    return cfg
