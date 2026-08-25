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


#: Heavy-tailed features (skew > ~1.5, max/median in the hundreds) that get a
#: log1p before standardisation so a net is not fitting a handful of +5-sigma rows.
LOG1P_FEATURES = ("accumulation", "rain_7", "rain_30", "rain_365", "slope")

#: Features that take exactly one value per station: terrain + soil. They are a
#: 37-row lookup table, not 50k samples -- the channel through which a net
#: memorises station identity. They get their own branch and their own noise.
STATIC_FEATURES = ("elevation", "slope", "northness", "eastness", "twi", "hli", "accumulation",
                   "soil_clay", "soil_sand", "soil_awc", "soil_bdw")


@dataclass(frozen=True)
class DataConfig:
    features: tuple[str, ...] = MODEL6_FEATURES
    target: str = TARGET
    group: str = GROUP
    time: str = TIME
    log1p: tuple[str, ...] = LOG1P_FEATURES       # () disables
    static: tuple[str, ...] = STATIC_FEATURES     # () = treat everything as dynamic

    @property
    def log_idx(self) -> tuple[int, ...]:
        return tuple(i for i, f in enumerate(self.features) if f in self.log1p)

    @property
    def static_idx(self) -> tuple[int, ...]:
        return tuple(i for i, f in enumerate(self.features) if f in self.static)


@dataclass(frozen=True)
class MLPConfig:
    hidden: tuple[int, ...] = (256, 256, 128)
    dropout: float = 0.15
    residual: bool = True
    static_bottleneck: int = 0      # >0: statics pass through a Linear->SiLU->Dropout of this width
    static_dropout: float = 0.3     # dropout inside the static branch


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
    input_noise: float = 0.05    # Gaussian noise on standardised DYNAMIC inputs
    static_noise: float = 0.3    # Gaussian noise on standardised STATIC inputs (blur the station fingerprint)
    val_frac: float = 0.15       # fraction of STATIONS held out for early stopping; 0 = off
    patience: int = 20
    n_ensemble: int = 3
    seed: int = 0
    amp: bool = True             # bf16 autocast on CUDA
    device: str | None = None    # None = cuda if available

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# sequence model (no SMIPS: SILO forcing window + station statics)
# --------------------------------------------------------------------------- #
FORCING_VARS = ("daily_rain", "et_morton_potential", "vp_deficit")
STATIC_VARS = ("soil_clay", "soil_sand", "soil_awc", "soil_bdw",
               "elevation", "slope", "northness", "eastness", "twi", "hli", "accumulation",
               "aridity")
LOG1P_STATIC = ("slope", "accumulation")


@dataclass(frozen=True)
class SeqDataConfig:
    forcing: tuple[str, ...] = FORCING_VARS
    statics: tuple[str, ...] = STATIC_VARS
    log1p_static: tuple[str, ...] = LOG1P_STATIC
    lookback: int = 365            # days of forcing before (and including) the target day
    target: str = TARGET
    group: str = GROUP
    time: str = TIME

    @property
    def static_log_idx(self) -> tuple[int, ...]:
        return tuple(i for i, f in enumerate(self.statics) if f in self.log1p_static)


@dataclass(frozen=True)
class TransformerConfig:
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    static_dropout: float = 0.3    # dropout on the static token embedding
    readout: str = "last"          # "last" token (= target day) | "mean" over tokens
