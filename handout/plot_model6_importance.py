"""model6 permutation feature importance, coloured by feature group.

Fits model6 once on the full 36-station table and computes permutation
importance (the drop in R^2/NSE when each feature is shuffled). Bars are grouped
by source so the SMIPS-lookback / terrain / soil / antecedent contributions are
legible.

Run from repo root::  PYTHONPATH=. python handout/plot_model6_importance.py
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from emt.model6 import model as m6
from emt.persist import fit_cached

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model6_importance.png"
TABLE = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
TARGET = m6.TARGET
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# feature -> group / colour
def group(f):
    if f == "smips_totalbucket" or f.startswith("smips_"):
        return "SMIPS (coarse + lookback)", "#1f77b4"
    if f.startswith("soil_"):
        return "SLGA soil", "#9467bd"
    if f.startswith(("rain_", "ppet_", "vpd_")):
        return "antecedent weather", "#d62728"
    if f.startswith("doy_"):
        return "seasonality", "#7f7f7f"
    return "terrain", "#2ca02c"

stamp("loading (or fitting once + saving) model6 ...")
sub = m6.ensure_features(pd.read_csv(TABLE)).dropna(subset=m6.FEATURES + [TARGET])
est = fit_cached(m6, sub, "model6")           # trained once, cached to data/models/
stamp("permutation importance (n_repeats=8) ...")
r = permutation_importance(est, sub[m6.FEATURES], sub[TARGET],
                           n_repeats=8, random_state=0, n_jobs=-1)
imp = pd.Series(r.importances_mean, index=m6.FEATURES).sort_values()
err = pd.Series(r.importances_std, index=m6.FEATURES).reindex(imp.index)
stamp("importance computed; plotting")

colors = [group(f)[1] for f in imp.index]
fig, ax = plt.subplots(figsize=(9, 9))
y = np.arange(len(imp))
ax.barh(y, imp.values, xerr=err.values, color=colors, error_kw=dict(alpha=.4))
ax.set_yticks(y); ax.set_yticklabels(imp.index, fontsize=9)
ax.set_xlabel("permutation importance (drop in NSE when shuffled)")
ax.set_title(f"model6 feature importance (36 stations, honest lookback features)\n"
             f"pooled leave-site-out NSE +0.38", fontsize=12)
# legend of groups
seen = {}
for f in m6.FEATURES:
    g, c = group(f); seen[g] = c
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in seen.values()]
ax.legend(handles, seen.keys(), fontsize=9, loc="lower right")
ax.grid(alpha=.3, axis="x")
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)
stamp(f"wrote {FIG.relative_to(REPO)}")
print("\ntop 10:\n" + imp.sort_values(ascending=False).head(10).round(4).to_string(), flush=True)
