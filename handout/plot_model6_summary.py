"""model6 4-panel results figure (like model4_results.png), from caches.

(a) leave-site-out fit; (b) permutation importance; (c) per-station NSE,
model4 -> model6; (d) per-station bias. Uses cached LOSO predictions
(data/*_loso_predictions.csv) and the saved model (data/models/model6.joblib),
so nothing re-fits once those exist.

Run from repo root::  PYTHONPATH=. python handout/plot_model6_summary.py
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

from emt.model4 import model as m4
from emt.model6 import model as m6
from emt.evaluation import metrics
from emt.persist import loso_cached, fit_cached

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model6_results.png"
TABLE = REPO / "data" / "train_catchment_plus_m_2006_2010.csv"
TARGET = m6.TARGET
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]
t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def grp(f):
    if f == "smips_totalbucket" or f.startswith("smips_"): return "#1f77b4"
    if f.startswith("soil_"): return "#9467bd"
    if f.startswith(("rain_", "ppet_", "vpd_")): return "#d62728"
    if f.startswith("doy_"): return "#7f7f7f"
    return "#2ca02c"


tab = pd.read_csv(TABLE)
site_of = tab.drop_duplicates("station").set_index("station")["site"]
stamp("model6 LOSO predictions (cached) ...")
p6 = loso_cached(m6, tab, "model6")
stamp("model4 LOSO predictions (cached; computes once) ...")
p4 = loso_cached(m4, tab, "model4")
p6["site"] = p6["station"].map(site_of)

def per_site(p):
    return p.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                      include_groups=False)
ps6, ps4 = per_site(p6), per_site(p4)
pooled6 = metrics(p6[TARGET], p6["pred"])

stamp("permutation importance (saved model) ...")
sub = m6.ensure_features(tab).dropna(subset=m6.FEATURES + [TARGET])
est = fit_cached(m6, sub, "model6")
r = permutation_importance(est, sub[m6.FEATURES], sub[TARGET], n_repeats=6,
                           random_state=0, n_jobs=-1)
imp = pd.Series(r.importances_mean, index=m6.FEATURES).sort_values()

fig, ax = plt.subplots(2, 2, figsize=(15, 11))

# (a) LOSO fit
for s in SITES:
    d = p6[p6.site == s]
    ax[0, 0].scatter(d[TARGET], d["pred"], s=6, alpha=.25, color=SITE_COLOR[s], label=s)
lim = [0, 55]
ax[0, 0].plot(lim, lim, "k--", lw=1)
ax[0, 0].set(xlim=lim, ylim=lim, xlabel="observed root-zone (%)", ylabel="LOSO predicted (%)",
             title=f"(a) model6 leave-site-out fit  (pooled NSE {pooled6['nse']:+.2f}, r {pooled6['r']:.2f})")
leg = ax[0, 0].legend(fontsize=9, markerscale=2)
for lh in leg.legend_handles: lh.set_alpha(1)
ax[0, 0].grid(alpha=.3)

# (b) importance
y = np.arange(len(imp))
ax[0, 1].barh(y, imp.values, color=[grp(f) for f in imp.index])
ax[0, 1].set_yticks(y); ax[0, 1].set_yticklabels(imp.index, fontsize=7)
ax[0, 1].set(xlabel="permutation importance", title="(b) Feature importance (soil dominates)")
ax[0, 1].grid(alpha=.3, axis="x")

# (c) per-station NSE, model4 -> model6
common = [s for s in ps4.index if s in ps6.index]
comp = pd.DataFrame({"m4": ps4.loc[common, "nse"], "m6": ps6.loc[common, "nse"]}).clip(-3, 1).sort_values("m6")
x = np.arange(len(comp))
ax[1, 0].vlines(x, comp["m4"], comp["m6"], color="#bbb", lw=1.2, zorder=1)
ax[1, 0].scatter(x, comp["m4"], s=22, color="#bbb", label="model4", zorder=2)
ax[1, 0].scatter(x, comp["m6"], s=26, color=[SITE_COLOR[site_of[s]] for s in comp.index],
                 edgecolor="k", linewidth=.3, label="model6", zorder=3)
ax[1, 0].axhline(0, color="k", lw=.8)
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(comp.index, rotation=90, fontsize=7)
ax[1, 0].set(ylim=[-3.2, 1.05], ylabel="per-station NSE",
             title=f"(c) Per-station NSE: model4 → model6  "
                   f"({int((ps4['nse']>0).sum())} → {int((ps6['nse']>0).sum())} positive)")
ax[1, 0].legend(fontsize=8, loc="lower right"); ax[1, 0].grid(alpha=.3, axis="y")

# (d) per-station bias
b = ps6.loc[common, "bias"].sort_values()
ax[1, 1].bar(range(len(b)), b.values, color=[SITE_COLOR[site_of[s]] for s in b.index])
ax[1, 1].axhline(0, color="k", lw=.8)
ax[1, 1].set_xticks(range(len(b))); ax[1, 1].set_xticklabels(b.index, rotation=90, fontsize=7)
ax[1, 1].set(ylabel="per-station bias (pred−obs, %)",
             title=f"(d) Residual per-station bias (median |bias| {ps6['bias'].abs().median():.1f}%)")
ax[1, 1].grid(alpha=.3, axis="y")

fig.suptitle("model6 (honest, leak-free), leave-site-out over 36 stations, 2006–2010",
             y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)
stamp(f"wrote {FIG.relative_to(REPO)}")
