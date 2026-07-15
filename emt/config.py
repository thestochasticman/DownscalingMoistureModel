"""Project paths and shared constants.

Data is cached under ``DATA_DIR`` (repo-local ``data/``, gitignored). Override
with the ``EMT_DATA_DIR`` environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("EMT_DATA_DIR", REPO_ROOT / "data"))
OZNET_DIR = DATA_DIR / "oznet"
SMIPS_DIR = DATA_DIR / "smips"
TERRAIN_DIR = DATA_DIR / "terrain"

for _d in (DATA_DIR, OZNET_DIR, SMIPS_DIR, TERRAIN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# OzNet Murrumbidgee catchment sites (excludes the JAXA flux site).
MURRUMBIDGEE_SITES = ("YANCO", "KYEAMBA", "MURRUMBIDGEE", "ADELONG")
