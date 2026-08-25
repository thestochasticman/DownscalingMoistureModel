"""Configuration dataclasses. Everything tunable lives here, nothing else."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

TARGET = "sm_rootzone_pct"
GROUP = "station"
TIME = "time"

#: model6's feature set, spelled out so this package does not import the
#: data-fetch stack (PaddockTS) just to learn a list of column names.
#: Order: coarse predictor, terrain, season, SMIPS lookback, soil, antecedent met.
MODEL6_FEATURES = (
    "smips_totalbucket",
    "elevation", "slope", "northness", "eastness", "twi", "hli", "accumulation",
    "doy_sin", "doy_cos",
    "smips_7d", "smips_30d", "smips_365d", "smips_anom",
    "soil_clay", "soil_sand", "soil_awc", "soil_bdw",
    "rain_7", "rain_30", "rain_365", "ppet_30", "ppet_365", "vpd_30", "rain_365_anom",
)


@dataclass(frozen=True)
class DataConfig:
    features: tuple[str, ...] = MODEL6_FEATURES
    target: str = TARGET
    group: str = GROUP
    time: str = TIME


@dataclass(frozen=True)
class MLPConfig:
    hidden: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.15
    residual: bool = True


@dataclass(frozen=True)
class TrainConfig:
    loss: str = "mse"            # "mse" | "huber" | "nse"
    nse_eps: float = 0.1         # NSE*: err^2 / (sigma_station + eps)^2, standardised units
    huber_delta: float = 1.0
    lr: float = 2e-3             # one-cycle peak
    weight_decay: float = 1e-3   # AdamW, decoupled
    epochs: int = 150
    batch_size: int = 1024
    warmup_frac: float = 0.1
    clip_grad: float | None = 1.0
    input_noise: float = 0.05    # Gaussian noise on standardised inputs
    val_frac: float = 0.15       # fraction of STATIONS held out for early stopping; 0 = off
    patience: int = 20
    n_ensemble: int = 3
    seed: int = 0
    amp: bool = True             # bf16 autocast on CUDA
    device: str | None = None    # None = cuda if available

    def to_dict(self) -> dict:
        return asdict(self)
