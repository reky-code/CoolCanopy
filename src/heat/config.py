"""Load the per-city YAML config."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["fq_schema"] = f"{cfg['catalog']}.{cfg['schema']}"
    return cfg
