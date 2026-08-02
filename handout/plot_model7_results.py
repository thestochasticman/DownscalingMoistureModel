"""model7 4-panel results figure, from cached LOSO predictions.

(a) leave-site-out fit (terrain-offset variant); (b) an example held-out
station's time series (the median-r station); (c) per-station NSE, base bucket
-> +terrain offsets; (d) per-station bias. Uses the cached LOSO prediction
tables (data/model7_loso_predictions.csv, data/model7t_loso_predictions.csv),
so nothing re-calibrates once those exist.

Run from repo root::  PYTHONPATH=. python handout/plot_model7_results.py
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emt.evaluation import metrics

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "handout" / "figures" / "model7_results.png"
TARGET = "sm_rootzone_pct"
SITE_COLOR = {"YANCO": "#1f77b4", "KYEAMBA": "#2ca02c", "ADELONG": "#d62728",
              "MURRUMBIDGEE": "#9467bd"}
SITES = ["ADELONG", "KYEAMBA", "MURRUMBIDGEE", "YANCO"]

tab = pd.read_csv(REPO / "data" / "process_target_2006_2010.csv")
site_of = tab.drop_duplicates("station").set_index("station")["site"]
p_base = pd.read_csv(REPO / "data" / "model7_loso_predictions.csv")
p_terr = pd.read_csv(REPO / "data" / "model7t_loso_predictions.csv")
p_terr["site"] = p_terr["station"].map(site_of)


def per_site(p):
    return p.groupby("station").apply(lambda g: pd.Series(metrics(g[TARGET], g["pred"])),
                                      include_groups=False)


ps_b, ps_t = per_site(p_base), per_site(p_terr)
pooled = metrics(p_terr[TARGET], p_terr["pred"])

fig, ax = plt.subplots(2, 2, figsize=(15, 11))

# (a) LOSO fit
for s in SITES:
    d = p_terr[p_terr.site == s]
    ax[0, 0].scatter(d[TARGET], d["pred"], s=6, alpha=.25, color=SITE_COLOR[s], label=s)
lim = [0, 55]
ax[0, 0].plot(lim, lim, "k--", lw=1)
ax[0, 0].set(xlim=lim, ylim=lim, xlabel="observed root-zone (%)", ylabel="LOSO predicted (%)",
             title=f"(a) model7 leave-site-out fit  (pooled NSE {pooled['nse']:+.2f}, r {pooled['r']:.2f})")
leg = ax[0, 0].legend(fontsize=9, markerscale=2)
for lh in leg.legend_handles: lh.set_alpha(1)
ax[0, 0].grid(alpha=.3)

# (b) example held-out time series: the station at the median per-station r
med_stn = ps_t["r"].sort_values().index[len(ps_t) // 2]
d = p_terr[p_terr.station == med_stn].copy()
d["time"] = pd.to_datetime(d["time"])
d = d.sort_values("time")
ax[0, 1].plot(d["time"], d[TARGET], color="k", lw=.9, label="observed")
ax[0, 1].plot(d["time"], d["pred"], color=SITE_COLOR[site_of[med_stn]], lw=.9,
              label="model7 (held out)")
m = metrics(d[TARGET], d["pred"])
ax[0, 1].set(ylabel="root-zone soil moisture (%)",
             title=f"(b) Held-out {med_stn} — the median-r station "
                   f"(r {m['r']:.2f}, NSE {m['nse']:+.2f})")
ax[0, 1].legend(fontsize=9)
ax[0, 1].grid(alpha=.3)

# (c) per-station NSE, base bucket -> +terrain offsets
common = [s for s in ps_b.index if s in ps_t.index]
comp = pd.DataFrame({"base": ps_b.loc[common, "nse"],
                     "terr": ps_t.loc[common, "nse"]}).clip(-3, 1).sort_values("terr")
x = np.arange(len(comp))
ax[1, 0].vlines(x, comp["base"], comp["terr"], color="#bbb", lw=1.2, zorder=1)
ax[1, 0].scatter(x, comp["base"], s=22, color="#bbb", label="bucket only", zorder=2)
ax[1, 0].scatter(x, comp["terr"], s=26, color=[SITE_COLOR[site_of[s]] for s in comp.index],
                 edgecolor="k", linewidth=.3, label="+ terrain offsets", zorder=3)
ax[1, 0].axhline(0, color="k", lw=.8)
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(comp.index, rotation=90, fontsize=7)
ax[1, 0].set(ylim=[-3.2, 1.05], ylabel="per-station NSE",
             title=f"(c) Per-station NSE: bucket → +terrain  "
                   f"({int((ps_b['nse']>0).sum())} → {int((ps_t['nse']>0).sum())} positive)")
ax[1, 0].legend(fontsize=8, loc="lower right"); ax[1, 0].grid(alpha=.3, axis="y")

# (d) per-station bias
b = ps_t.loc[common, "bias"].sort_values()
ax[1, 1].bar(range(len(b)), b.values, color=[SITE_COLOR[site_of[s]] for s in b.index])
ax[1, 1].axhline(0, color="k", lw=.8)
ax[1, 1].set_xticks(range(len(b))); ax[1, 1].set_xticklabels(b.index, rotation=90, fontsize=7)
ax[1, 1].set(ylabel="per-station bias (pred−obs, %)",
             title=f"(d) Residual per-station bias (median |bias| {ps_t['bias'].abs().median():.1f}%)")
ax[1, 1].grid(alpha=.3, axis="y")

fig.suptitle("model7 (process model: calibrated bucket water balance), "
             "leave-site-out over 37 stations, 2006–2010", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG, dpi=130)
plt.close(fig)
print(f"wrote {FIG.relative_to(REPO)}")
