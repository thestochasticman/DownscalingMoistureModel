"""Global configuration for the Water Balance Model.

Configuration is loaded from ``~/.config/WaterBalanceModel.json`` if it exists,
otherwise defaults are used. The config object is a frozen attrs class.

Example config file::

    {
        "out_dir": "/data/water_balance/outputs",
        "tmp_dir": "/data/water_balance/tmp",
        "email": "user@example.com",
        "tern_api_key": "your-tern-key"
    }
"""

from __future__ import annotations

import os
from json import load
from os import makedirs
from os.path import exists, expanduser
from typing import Optional

from attrs import frozen
from typing_extensions import Self


@frozen
class Config:
    """Global configuration for the Water Balance Model.

    Attributes:
        out_dir: Directory for final outputs (Zarr stores, plots).
        tmp_dir: Directory for intermediate cached data.
        email: SILO API registration email.
        tern_api_key: TERN API key for SMIPS access.
    """

    out_dir: str
    tmp_dir: str
    email: Optional[str] = None
    tern_api_key: Optional[str] = None

    def __attrs_post_init__(s: Self) -> None:
        makedirs(s.out_dir, exist_ok=True)
        makedirs(s.tmp_dir, exist_ok=True)


_out = expanduser('~/Documents/WaterBalanceModel-Outputs')
_tmp = expanduser('~/Downloads/WaterBalanceModel-Tmp')
_default = Config(_out, _tmp, None, None)

confpath = expanduser('~/.config/WaterBalanceModel.json')
config: Config = Config(**load(open(confpath))) if exists(confpath) else _default
