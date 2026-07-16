"""Project paths and shared constants.

Repo-specific data lives under ``DATA_DIR`` (repo-local ``data/``): the training
tables, trained models, LOSO predictions and the OzNet in-situ cache. Override
with the ``EMT_DATA_DIR`` environment variable. Everything downloaded via
PaddockTS (SMIPS, terrain, SLGA soil, SILO) is cached in the per-AOI PaddockTS
query tree (``query.tmp_dir``/``query.out_dir``), not here.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("EMT_DATA_DIR", REPO_ROOT / "data"))
OZNET_DIR = DATA_DIR / "oznet"      # OzNet in-situ .xls cache (EMT-fetched)

for _d in (DATA_DIR, OZNET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# OzNet Murrumbidgee catchment sites (excludes the JAXA flux site).
MURRUMBIDGEE_SITES = ("YANCO", "KYEAMBA", "MURRUMBIDGEE", "ADELONG")
