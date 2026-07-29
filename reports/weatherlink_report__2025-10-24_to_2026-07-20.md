# WeatherLink profile-mean point-only validation

Sensor lsid: `591644`  
Profile-mean CSV: `weatherlink_validation/outputs/profile_mean_591644_2025-07-21_2026-07-20/download/weatherlink_591644_profile_mean_generic_model6_validation.csv`  
Date range: 2025-10-24 to 2026-07-20  
Observations: 268

Only the profile-mean Drill & Drop value is used. Individual depths are not used
as validation points.

## Pooled model6 skill

| Metric | Value |
|---|---:|
| NSE/R² | 0.570 |
| Pearson r | 0.936 |
| RMSE | 3.473 % |
| ubRMSE | 3.274 % |
| Bias | 1.160 % |
| n | 268 |

## Trailing-window comparison

| window | start_date | end_date | n | nse | r | rmse | bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| last_20_days | 2026-07-01 | 2026-07-20 | 20 | -121.283 | 0.520 | 3.649 | -3.480 |
| last_45_days | 2026-06-06 | 2026-07-20 | 45 | -24.709 | 0.818 | 4.007 | -3.912 |
| last_60_days | 2026-05-22 | 2026-07-20 | 60 | -1.143 | 0.814 | 3.769 | -3.435 |
| last_90_days | 2026-04-22 | 2026-07-20 | 89 | 0.560 | 0.939 | 3.286 | -1.704 |
| last_120_days | 2026-03-23 | 2026-07-20 | 118 | 0.643 | 0.950 | 2.969 | -0.894 |
| last_180_days | 2026-01-22 | 2026-07-20 | 178 | 0.624 | 0.962 | 3.549 | 0.776 |
| last_270_days | 2025-10-24 | 2026-07-20 | 268 | 0.570 | 0.936 | 3.473 | 1.160 |
| last_365_days | 2025-10-24 | 2026-07-20 | 268 | 0.570 | 0.936 | 3.473 | 1.160 |
| all | 2025-10-24 | 2026-07-20 | 268 | 0.570 | 0.936 | 3.473 | 1.160 |

## Temporal self-spiking

Spatial spiking is skipped because this is one profile. Temporal self-spiking
uses the first N profile-mean observations to estimate a simple local bias
correction, then evaluates later observations.

| training_dates | test_observations | correction_pct | nse | baseline_nse | delta_nse | rmse | baseline_rmse | delta_rmse | bias |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 267.000 | 3.111 | -0.033 | 0.571 | -0.604 | 5.391 | 3.474 | 1.917 | 4.286 |
| 2.000 | 266.000 | 3.132 | -0.041 | 0.571 | -0.613 | 5.417 | 3.475 | 1.942 | 4.323 |
| 3.000 | 265.000 | 2.998 | -0.003 | 0.572 | -0.575 | 5.322 | 3.478 | 1.844 | 4.205 |
| 5.000 | 263.000 | 2.811 | 0.048 | 0.572 | -0.524 | 5.194 | 3.484 | 1.710 | 4.046 |
| 10.000 | 258.000 | 2.798 | 0.043 | 0.573 | -0.530 | 5.235 | 3.496 | 1.739 | 4.111 |
| 20.000 | 248.000 | 2.178 | 0.201 | 0.573 | -0.372 | 4.855 | 3.551 | 1.305 | 3.607 |
| 30.000 | 238.000 | 1.427 | 0.368 | 0.573 | -0.204 | 4.404 | 3.623 | 0.782 | 2.912 |
| 60.000 | 208.000 | -0.660 | 0.628 | 0.589 | 0.039 | 3.522 | 3.700 | -0.178 | 0.644 |
| 90.000 | 178.000 | -1.919 | 0.603 | 0.624 | -0.021 | 3.647 | 3.549 | 0.098 | -1.144 |
| 120.000 | 148.000 | -2.784 | 0.332 | 0.654 | -0.322 | 4.229 | 3.041 | 1.187 | -2.942 |

## Interpretation note

If NSE improves in longer windows, the short-window negative NSE was likely
being driven by low observed variance over the 20-day winter subset. If NSE
remains negative despite stronger correlation or low RMSE, model6 is capturing
some level/direction information but not the full local temporal dynamics.

Because the current temporal self-spiking is only a simple bias correction, not a real retraining of the model. It estimates 

>correction = mean(observed - model6 prediction) over first N days\
>corrected prediction = model6 prediction + correction

For this sensor, model6 error is strongly seasonal, not a constant offset.

From the long profile-mean run:\
- `Oct 2025: model6 underpredicts by about +2.82% obs-minus-pred.`\
- `Dec–Feb: model6 overpredicts by about -3.6% to -5.3%.`\
- `Jun–Jul: model6 underpredicts again by about +3.5% to +4.2%.`

So if the first few training days are in Oct/Nov, the calibration learns a positive correction. 
But much of the later test period needs a negative correction, so the “local calibration” pushes 
predictions the wrong way and worsens RMSE/NSE.

Also, the uncalibrated model6 already performs well over the full usable 268-day record:

>NSE/R² = 0.570\
>r      = 0.936\
>RMSE   = 3.47%

So a crude intercept-only correction has little room to help and plenty of room to damage the seasonal shape.
One exception: with 60 training days, temporal spiking slightly helps:

>baseline NSE = 0.589\
>spiked NSE   = 0.628\
>baseline RMSE = 3.70%\
>spiked RMSE   = 3.52%

That’s because the 60-day correction is closer to the later-period bias. But with 90 or 120 training days, it overcorrects again.

Seasonal bias correction is also likely an issue with the dense points validation, which is obscured by the limited data collection window. 
To detect whether the dense training dataset local tuning performs well in other seasons we can utilise the in-situ soil moisture probes. 

## Figures:

- `figures/timeseries_obs_vs_model6.png`
- `figures/scatter_obs_vs_model6.png`
- `figures/nse_by_trailing_window.png`
- `figures/cumulative_nse.png`

## Profile metadata

| node_name | lsid | depth_min_cm | depth_max_cm | n_depths |
| --- | --- | --- | --- | --- |
| Kowen High 90 cm Basalt  | 591644 | 10.000 | 90.000 | 9 |
