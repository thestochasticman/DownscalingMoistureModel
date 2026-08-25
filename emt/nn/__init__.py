"""Neural-network track.

    data      -- TabularData: features / target / station / time, standardisation, grouped splits
    losses    -- mse, huber, nse (per-station NSE*)
    mlp       -- ResidualMLP
    train     -- Trainer: one net, one config, one training run
    model     -- MLPModel: ensemble of trained nets with save/load and DataFrame I/O
    cv        -- the validation ladder (station / block / year / blockyear)
    config    -- the dataclasses that parametrise all of the above
"""
from emt.nn.config import DataConfig, MLPConfig, TrainConfig
from emt.nn.data import TabularData
from emt.nn.model import MLPModel

__all__ = ["DataConfig", "MLPConfig", "TrainConfig", "TabularData", "MLPModel"]
