"""Project paths and shared constants.

Data is cached under ``DATA_DIR`` (repo-local ``data/``, gitignored). Generated
maps are written under ``OUTPUTS_DIR`` (repo-local ``outputs/``, gitignored).
Override either with the ``EMT_DATA_DIR`` / ``EMT_OUTPUTS_DIR`` environment
variables.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("EMT_DATA_DIR", REPO_ROOT / "data"))
OZNET_DIR = DATA_DIR / "oznet"
SMIPS_DIR = DATA_DIR / "smips"
TERRAIN_DIR = DATA_DIR / "terrain"

# Generated products (downscaled GeoTIFFs) — the predict CLI writes here.
OUTPUTS_DIR = Path(os.environ.get("EMT_OUTPUTS_DIR", REPO_ROOT / "outputs"))

for _d in (DATA_DIR, OZNET_DIR, SMIPS_DIR, TERRAIN_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# OzNet Murrumbidgee catchment sites (excludes the JAXA flux site).
MURRUMBIDGEE_SITES = ("YANCO", "KYEAMBA", "MURRUMBIDGEE", "ADELONG")
