#!/usr/bin/env python3
"""Build the canonical OzNet training table and retrain EMT model6.

This is intentionally a thin orchestration layer over the existing framework:

* ``emt.build_dataset.build`` creates the OzNet + SMIPS + terrain + soil table.
* ``emt.model6.model.ensure_features`` adds model6's SMIPS lookbacks and SILO
  antecedent-meteorology predictors.
* ``emt.model6.model.fit`` trains the persisted model6 estimator.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from attrs import evolve


DEFAULT_DATA_DIR = Path("/Volumes/Dmitry_work/borevitz_projects/Data")
DEFAULT_SILO_EMAIL = "dmitry.grishin@anu.edu.au"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_process_only_silo_email(email: str | None) -> str:
    """Make SILO downloads work without editing ``~/.config/PaddockTS.json``.

    PaddockTS prefers its config file when present. If that file already carries
    an email, this function leaves it alone. If no email is configured, the
    fallback is installed only in this Python process.
    """
    if not email:
        return "no fallback email supplied"
    os.environ.setdefault("PADDOCKTS_EMAIL", email)

    import PaddockTS.config as pts_config

    if getattr(pts_config.config, "email", None):
        return "using existing PaddockTS SILO email/config"

    patched = evolve(pts_config.config, email=email)
    pts_config.config = patched
    return f"using process-only SILO fallback email {email}"


def _metric_line(m: dict) -> str:
    parts = []
    for k in ("rmse", "ubrmse", "bias", "r", "nse", "r2"):
        v = m.get(k)
        parts.append(f"{k}={v:.4f}" if np.isfinite(v) else f"{k}=nan")
    parts.append(f"n={int(m.get('n', 0))}")
    return ", ".join(parts)


def _write_report(
    *,
    path: Path,
    base_out: Path,
    train_out: Path,
    model_path: Path,
    loso_pred_out: Path | None,
    loso_site_out: Path | None,
    meta_out: Path,
    table: pd.DataFrame,
    fitted_table: pd.DataFrame,
    params: dict,
    in_sample: dict,
    loso: dict | None,
    silo_status: str,
) -> None:
    site_counts = {
        str(k): int(v)
        for k, v in table.groupby("site")["station"].nunique().sort_index().items()
    }
    feature_cols = [c for c in fitted_table.columns if c not in table.columns]
    lines = [
        "# EMT model6 OzNet retrain",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "This run uses the existing EMT model6 framework: the canonical OzNet "
        "builder plus `emt.model6.model.ensure_features()` and "
        "`emt.model6.model.fit()`.",
        "",
        "## Outputs",
        "",
        f"- Base OzNet table: `{base_out}`",
        f"- Model6-ready OzNet table: `{train_out}`",
        f"- Persisted model used by `emt.predict`: `{model_path}`",
        f"- Metadata JSON: `{meta_out}`",
    ]
    if loso_pred_out and loso_site_out:
        lines.extend([
            f"- Leave-station-out predictions: `{loso_pred_out}`",
            f"- Leave-station-out per-station metrics: `{loso_site_out}`",
        ])

    lines.extend([
        "",
        "## Dataset",
        "",
        f"- Base rows: {len(table):,}",
        f"- Model6-ready fitted rows: {len(fitted_table):,}",
        f"- Stations: {table['station'].nunique()}",
        f"- Sites: {site_counts}",
        f"- Date range: {pd.to_datetime(table['time']).min().date()} to "
        f"{pd.to_datetime(table['time']).max().date()}",
        f"- Added model6 feature columns: {feature_cols}",
        f"- SILO configuration: {silo_status}",
        "",
        "## Estimator",
        "",
        "```json",
        json.dumps(params, indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Training-table performance",
        "",
        f"- In-sample: {_metric_line(in_sample)}",
    ])
    if loso is not None:
        pooled = loso["pooled"]
        per_site = loso["per_site"]
        lines.extend([
            f"- Leave-station-out pooled: {_metric_line(pooled)}",
            f"- Leave-station-out stations with NSE > 0: "
            f"{int((per_site['nse'] > 0).sum())}/{len(per_site)}",
            f"- Leave-station-out median station NSE: {per_site['nse'].median():.4f}",
            f"- Leave-station-out median |bias|: {per_site['bias'].abs().median():.4f}",
        ])

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-name", default="model6")
    parser.add_argument("--max-leaf-nodes", type=int, default=255)
    parser.add_argument("--silo-email", default=DEFAULT_SILO_EMAIL)
    parser.add_argument("--skip-loso", action="store_true",
                        help="Fit and save the model without leave-station-out CV.")
    args = parser.parse_args()

    # Numba can fail to create caches for some installed package locations when
    # launched non-interactively. Keep cache writes in a writable temp folder.
    os.environ.setdefault("NUMBA_CACHE_DIR", "/private/tmp/numba_cache")

    silo_status = _install_process_only_silo_email(args.silo_email)
    print(f"SILO: {silo_status}", flush=True)

    from emt.build_dataset import DEFAULT_END, DEFAULT_START, build
    from emt.evaluation import leave_site_out_cv, metrics
    from emt.insitu.coordinates import COORDS_CACHE, fetch_station_coords
    from emt.insitu.oznet import fetch_manifest
    from emt.model6 import model as model6
    from emt.persist import _path, save_model

    args.data_dir.mkdir(parents=True, exist_ok=True)
    base_out = args.data_dir / "oznet_model6_base_training_2006_2010.csv"
    train_out = args.data_dir / "oznet_model6_training_2006_2010.csv"
    loso_pred_out = args.data_dir / "oznet_model6_loso_predictions_2006_2010.csv"
    loso_site_out = args.data_dir / "oznet_model6_loso_per_station_2006_2010.csv"
    report_out = args.data_dir / "oznet_model6_retrain_summary.md"
    meta_out = args.data_dir / "oznet_model6_retrain_metadata.json"

    print("Fetching OzNet manifest and station coordinates ...", flush=True)
    manifest = fetch_manifest()
    stations = sorted(manifest["station"].unique())
    coords = fetch_station_coords(stations, cache=COORDS_CACHE, refresh=False)
    n_coords = int(coords["has_coords"].sum())
    print(f"Coordinate cache: {COORDS_CACHE} ({n_coords}/{len(coords)} stations)", flush=True)

    print("Building canonical OzNet base training table ...", flush=True)
    table = build(start=DEFAULT_START, end=DEFAULT_END, out=str(base_out), verbose=True)

    print("Adding model6 feature columns ...", flush=True)
    full = model6.ensure_features(table)
    full.to_csv(train_out, index=False)
    print(f"Saved model6-ready table: {train_out} ({len(full):,} rows)", flush=True)

    fitted = full.dropna(subset=list(model6.FEATURES) + [model6.TARGET]).copy()
    print(f"Fitting model6 on {len(fitted):,} complete rows ...", flush=True)
    estimator = model6.build_estimator(max_leaf_nodes=args.max_leaf_nodes)
    model = model6.fit(full, estimator=estimator)
    model_path = save_model(model, args.model_name)
    print(f"Saved persisted model: {model_path}", flush=True)

    pred = model.predict(fitted[list(model6.FEATURES)])
    in_sample = metrics(fitted[model6.TARGET], pred)
    print(f"In-sample: {_metric_line(in_sample)}", flush=True)

    loso = None
    if not args.skip_loso:
        print("Running leave-station-out CV with the same model6 estimator ...", flush=True)
        loso = leave_site_out_cv(
            full,
            list(model6.FEATURES),
            lambda: model6.build_estimator(max_leaf_nodes=args.max_leaf_nodes),
            group_col="station",
            target=model6.TARGET,
        )
        loso["predictions"].to_csv(loso_pred_out, index=False)
        loso["per_site"].to_csv(loso_site_out, index=False)
        print(f"LOSO pooled: {_metric_line(loso['pooled'])}", flush=True)

    params = model.get_params()
    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "base_table": str(base_out),
        "model6_ready_table": str(train_out),
        "model_path": str(model_path),
        "model_path_absolute": str(_path(args.model_name).resolve()),
        "model_name": args.model_name,
        "estimator_params": params,
        "features": list(model6.FEATURES),
        "target": model6.TARGET,
        "base_rows": int(len(table)),
        "fitted_rows": int(len(fitted)),
        "stations": int(table["station"].nunique()),
        "sites": {
            str(k): int(v)
            for k, v in table.groupby("site")["station"].nunique().to_dict().items()
        },
        "in_sample": in_sample,
        "loso_pooled": loso["pooled"] if loso is not None else None,
        "silo_status": silo_status,
    }
    meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")

    _write_report(
        path=report_out,
        base_out=base_out,
        train_out=train_out,
        model_path=model_path,
        loso_pred_out=loso_pred_out if loso is not None else None,
        loso_site_out=loso_site_out if loso is not None else None,
        meta_out=meta_out,
        table=table,
        fitted_table=fitted,
        params=params,
        in_sample=in_sample,
        loso=loso,
        silo_status=silo_status,
    )
    print(f"Saved report: {report_out}", flush=True)

    # Also keep an archive copy on the requested volume next to the training
    # table; the canonical runtime model remains data/models/model6.joblib.
    archive_model = args.data_dir / f"{args.model_name}_oznet_retrained_leaves{args.max_leaf_nodes}.joblib"
    joblib.dump(model, archive_model)
    print(f"Saved archive model copy: {archive_model}", flush=True)


if __name__ == "__main__":
    main()
