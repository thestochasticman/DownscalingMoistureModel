# Progress report: downscaling soil moisture to paddock scale for management and ecophysiology modelling

Date: 2026-07-25  
Project working folder: `/Volumes/Dmitry_work/borevitz_projects`  
Repository: `/Volumes/Dmitry_work/borevitz_projects/DownscalingMoistureModel`  
Current branch: `EMT`

## Executive summary

This work develops and tests a statistical downscaling pipeline that turns coarse
Australian soil-moisture information into daily sub-property / paddock-scale
predictions suitable for management decisions and ecophysiology modelling. The
current implementation predicts a 30 m root-zone soil-moisture field from public
or national gridded inputs: TERN SMIPS soil water, terrain derivatives, SLGA soil
properties, SILO antecedent meteorology and seasonality.

The chosen model, **model6**, is a regularised histogram gradient-boosting model.
It was selected because it gave the best combination of held-out skill,
interpretable hydrological inputs, national applicability and manageable
computation. Its design is deliberately conservative: station coordinates are
not predictors, and dynamic predictors are computed as backward-looking
lookbacks to avoid leakage.

The model performs reasonably in the original OzNet/Murrumbidgee leave-site-out
setting, but external dense-point validation shows a harder story. Against a new
high-density point dataset, the uncalibrated shipped model6 had weak pooled
skill (`NSE/R² = 0.023`, `r = 0.238`), although 43 of 79 individual points had
positive NSE/R². The dense dataset is therefore valuable not because it confirms
the current model as finished, but because it reveals where the transfer breaks:
terrain extremes such as high TWI, high HLI/northness contexts and high
long-term SMIPS wetness are associated with poorer point-level performance.

Local training-data experiments show that this dense dataset contains strong
calibration signal. Local residual calibration and direct local model6 retraining
both substantially improve point-level accuracy. A direct local retrain of
model6-style gradient boosting with a larger leaf budget reached a point-held-out
GroupKFold `NSE/R² ≈ 0.704`, and an in-sample final dense-site fit reached
`NSE/R² ≈ 0.959`. This is promising, but it should be interpreted as local
calibration performance, not proof of broader generalisation.

## 1. Challenge and project framing

Soil moisture controls plant water stress, stomatal conductance, growth,
carbon-water trade-offs, rainfall response, runoff generation and recovery after
dry periods. For management and ecophysiology, the most useful scale is often
sub-property: a paddock, restoration block, trial plot, gully line, ridge or
vegetation patch. Unfortunately, the strongest public soil-moisture products are
usually coarser than that. Microwave and modelled products provide valuable
temporal information, but their native resolution is generally too coarse to
represent within-property terrain, aspect, soil and drainage variation.

The core challenge is therefore a spatial disaggregation problem:

> Given a coarse daily estimate of soil water, where should the wetter and drier
> 30 m pixels be within the paddock-scale landscape?

This project frames that as a learnable redistribution problem. SMIPS provides
the coarse hydrological state; terrain, soil and recent meteorology provide the
within-pixel redistribution structure. The aim is not only to create maps, but
to create maps with enough local credibility to support management and
ecophysiology modelling.

## 2. Model development pathway

The model sequence developed in the handout progresses from simple baselines to
the chosen model6.

![model6 held-out performance](../handout/figures/model6_results.png)

### 2.1 Model family

The models were developed in stages:

1. **model1** — baseline random-forest-style model using coarse SMIPS, terrain
   and seasonal terms.
2. **model2** — simpler linear/interpretable benchmark.
3. **model3** — gradient boosting reference model.
4. **model4** — key improvement: adds SMIPS lookback/climatology terms and SLGA
   root-zone soil properties.
5. **model5** — tests soil smoothing; retained as a diagnostic rather than a
   selected product.
6. **model6** — chosen model: model4 plus antecedent meteorology from SILO.

Model6 uses:

- SMIPS total bucket and SMIPS lookback/anomaly terms;
- 30 m terrain: elevation, slope, northness, eastness, TWI, HLI and
  accumulation;
- SLGA soil: clay, sand, available water capacity and bulk density;
- SILO antecedent weather: rainfall, P−PET, VPD and 365 d rainfall anomaly;
- seasonality: cyclic day-of-year terms.

### 2.2 Why model6 is currently chosen

Model6 is the selected current model because it balances performance,
transferability and operational feasibility:

- It improves on earlier models in the original leave-site-out validation.
- It uses nationally available inputs, so the model can run anywhere in
  Australia where the covariates are available.
- It avoids using station coordinates as predictors, reducing the risk of
  memorising training stations.
- Its dynamic predictors are backward-looking, so prediction for a day does not
  use future information.
- It incorporates hydrologically meaningful controls: local terrain, soil water
  holding capacity, recent rainfall and evaporative demand.

The handout reports approximately:

| Skill, OzNet/Murrumbidgee leave-site-out | model6 |
|---|---:|
| Pooled NSE / r | `0.39 / 0.64` |
| Median per-station \|bias\| | `3.16 %` |
| Positive-NSE stations | `17/36` |

![model6 downscaled example](../handout/figures/downscale_gallery_model6.png)

## 3. Current limitations

The model is promising, but it is not finished.

### 3.1 Training-domain limitation

The base model is trained and validated primarily against OzNet/Murrumbidgee
root-zone observations. The Murrumbidgee Soil Moisture Monitoring Network is a
major Australian resource; Smith et al. describe the network as spanning the
82,000 km² Murrumbidgee catchment and including multiple soil-moisture depths,
temperature, precipitation and forcing data. That makes it valuable for remote
sensing and land-surface model validation, but it is still geographically and
sensor-context limited relative to a national operational product [Smith et al.,
2012](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012WR011976).

### 3.2 Target-depth limitation

The base model target is OzNet-style root-zone soil moisture, while the dense
point dataset appears to represent a shallower measurement. This mismatch is
important. A shallow handheld or near-surface device may respond more strongly
to recent rain, evaporation, litter, shading and microtopography than a 0–90 cm
root-zone average.

**Human fill-in:** confirm the dense-device measurement depth, calibration
equation and whether the intended operational product should be surface,
root-zone or both.

### 3.3 Bias and amplitude limitation

The dense validation shows that model6 often tracks some temporal structure but
compresses local spatial amplitude. Predictions vary less than point
observations. That means NSE/R² can be poor even when `r` is moderate, because
the model has a persistent level bias or insufficient local contrast.

### 3.4 Terrain-extreme limitation

The dense point validation suggests poorer transfer in high-TWI, high-HLI and
high-northness contexts. These are exactly the places where paddock-scale
ecophysiology may care most: wet drainage lines, exposed slopes, protected
microsites and local redistribution zones.

### 3.5 Missing uncertainty

The current output is a deterministic map. Operational management would benefit
from uncertainty maps, particularly in terrain contexts or soil states where the
training data are sparse.

## 4. Dense point validation

The dense point dataset was used as an independent external test. The workflow
generated model6 maps for all dates in the CSV, sampled predictions at each
point/date, computed handout-style metrics and then mapped point-level quality.

Validation output folder:

`/Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking`

### 4.1 Stage 1 — unseen high-density validation

Dataset summary:

- source rows: `631`;
- usable observations: `560`;
- excluded rows: `71`, mostly missing coordinates;
- georeferenced points: `79`;
- dates: `9`;
- period: `2025-04-30` to `2025-07-17`.

Pooled uncalibrated model6 performance:

| metric | value |
|---|---:|
| NSE / R² | `0.023` |
| Pearson r | `0.238` |
| RMSE | `6.477` |
| ubRMSE | `6.453` |
| bias | `-0.558 %` |
| n | `560` |

Per-point:

- positive NSE/R²: `43/79`;
- very poor NSE/R² (≤ -1): `13/79`;
- median |bias|: `2.26 %`.

![dense validation model-input correlations](/Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking/Validation_2stage/stage1_dense_unseen_validation/figures/nse_model_input_correlations.png)

The strongest model-input associations with point-level NSE/R² were:

| model input | Pearson r with point NSE/R² |
|---|---:|
| TWI | `-0.330` |
| northness | `-0.324` |
| SMIPS 365 d | `-0.318` |
| slope | `0.306` |
| soil bulk density | `0.282` |
| SMIPS anomaly | `0.264` |
| HLI | `-0.257` |

This implies that the model is not failing uniformly. It transfers better in
some parts of model-input space than others. High-TWI and high-HLI/northness
associations should be treated as diagnostic flags rather than causal
interpretations. TWI is a physically meaningful terrain index with a long
history in hydrological modelling: TOPMODEL relates hydrological similarity and
wetness tendency to upslope contributing area and slope, so it is unsurprising
that terrain wetness indices appear in both process models and statistical
downscaling [Beven and Kirkby, 1979](https://hero.epa.gov/reference/3349401).

Meteorological predictors were present but weaker in point-level spatial
correlations. This likely reflects the site scale: the nine dates share much of
the same antecedent weather signal across nearby points, while terrain and soil
vary more within the site.

![NSE vs selected model inputs](/Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking/Validation_2stage/stage1_dense_unseen_validation/figures/nse_vs_selected_model_inputs.png)

### 4.2 Stage 2 — sensitivity to local training-data spiking

Stage 2 tested whether supplying local observations improves prediction at
specific locations. Two experiments were run:

1. **Spatial spiking** — for each target point, hold that point out, then add
   increasing numbers of other local points as calibration data.
2. **Temporal self-spiking** — use early observations at a point to improve
   later predictions at that same point.

The first implementation used local residual calibration on top of shipped
model6. Best spatial spiking settings were:

| strategy | method | spike points | median ΔNSE/R² | median ΔRMSE |
|---|---|---:|---:|---:|
| terrain-stratified | ridge residual | 40 | `+0.695` | `-2.771` |
| terrain-similar | ridge residual | 40 | `+0.695` | `-2.770` |
| nearest | ridge residual | 40 | `+0.678` | `-2.848` |

![spatial spiking ridge learning curve](/Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking/Validation_2stage/stage2_local_spiking_sensitivity/figures/spatial_spiking_delta_nse_ridge_residual.png)

The ridge residual corrector was useful analytically, but when applied across
the whole map it extrapolated too strongly in some pixels. A smoother KNN
residual correction was therefore generated as the safer local-calibration map.

KNN local residual map point-level fit, in sample:

| metric | untrained model6 | KNN residual-trained |
|---|---:|---:|
| NSE / R² | `0.023` | `0.648` |
| r | `0.238` | `0.819` |
| RMSE | `6.477` | `3.890` |
| bias | `-0.558 %` | `0.032 %` |

### 4.3 Local model6 retraining with more leaves

A direct dense-site model6 retrain was then run using the dense point feature
table. The retrained model uses the same model6 inputs, but is fitted directly
to the dense local observations.

Selected local settings:

| parameter | value |
|---|---:|
| max_leaf_nodes | `511` |
| min_samples_leaf | `20` |
| max_iter | `300` |
| max_features | `0.3` |

The leaf sweep found that `127`, `255`, `511` and unlimited leaves were
effectively identical at fixed `min_samples_leaf`. In other words, this dense
dataset is not leaf-count limited; it is sample-support limited.

Point-held-out GroupKFold estimate:

| metric | value |
|---|---:|
| NSE / R² | `0.704` |
| r | `0.841` |
| RMSE | `3.565` |
| bias | `0.113 %` |

Final in-sample dense-site model fit:

| metric | value |
|---|---:|
| NSE / R² | `0.959` |
| r | `0.980` |
| RMSE | `1.320` |

![local model6 more-leaves comparison](/Volumes/Dmitry_work/borevitz_projects/model6_dense_validation_spiking/SM_comparison_trained_untrained/model6_more_leaves_dense_points/figures/SM_comparison_model6_more_leaves_2025-05-21.png)

This result is encouraging: the dense site contains strong local information.
However, it also warns that local retraining can create a site-specific product.
The next research question is not whether the dense data can improve a local
map — it clearly can — but how much of that improvement transfers to other
properties, seasons and measurement devices.

## 5. Other model classes

### 5.1 Classical machine-learning/statistical downscaling

The current approach sits within a broader literature of statistical and
machine-learning downscaling of satellite or modelled soil moisture. Reviews
describe common strategies that use ancillary variables such as vegetation,
surface temperature, topography and climate to improve the spatial resolution of
coarse soil-moisture products [Peng et al.,
2017](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016RG000543);
[Remote Sensing review,
2024](https://www.mdpi.com/2072-4292/16/12/2067). Similar work has used
geomorphometry and machine learning to downscale coarse soil-moisture products,
emphasising the value of topographic predictors
([study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6759172/)).

Pros in this context:

- works with available public gridded predictors;
- computationally cheap enough for daily paddock-scale runs;
- can learn non-linear terrain/soil/weather interactions;
- easy to update as more training data become available.

Cons:

- transfer depends on training coverage;
- can learn spurious correlations;
- weak physical guarantees, e.g. no automatic water-balance conservation;
- requires careful leakage control and spatial validation.

### 5.2 Optical/thermal and microwave fusion

Many downscaling approaches combine coarse microwave soil moisture with optical
or thermal indicators such as NDVI, land-surface temperature or SWIR water
indices. Piles et al. used MODIS visible/infrared information to downscale
SMOS-derived soil moisture
([Piles et al., 2011](https://www.documentation.ird.fr/hor/PAR00007912)).
Other studies use LST-NDVI or SWIR relationships to move from coarse passive
microwave estimates toward finer maps
([example](https://www.sciencedirect.com/science/article/pii/S002216941300944X)).

Pros:

- adds vegetation condition and surface-energy information;
- may help shallow/surface moisture targets;
- Sentinel/Landsat/S2/S1 products could support high-resolution mapping.

Cons:

- cloud, vegetation cover, overpass timing and scale mismatch matter;
- optical/thermal signals may represent surface moisture rather than root-zone
  moisture;
- requires careful temporal alignment.

### 5.3 AI / deep learning models

Neural networks and deep learning have been used for global and regional soil
moisture downscaling. For example, neural networks have been used to downscale
SMAP observations to finer scales
([HESS, 2018](https://hess.copernicus.org/articles/22/5341/2018/)).
Potential future models could include multilayer perceptrons, CNN/U-Net models,
spatio-temporal transformers or graph neural networks.

Pros:

- can learn complex non-linear interactions;
- CNN/U-Net architectures can exploit spatial context;
- transformers or sequence models could learn temporal memory;
- useful once much larger training datasets are assembled.

Cons:

- needs more training data than currently available;
- harder to interpret;
- higher risk of overfitting dense local sites;
- more difficult to enforce physical constraints and uncertainty;
- training/serving complexity increases.

### 5.4 Physics-based and process models

Physics-based approaches include Richards-equation soil-water models, HYDRUS,
land-surface models and TOPMODEL-like topographic approaches. HYDRUS and related
models simulate variably saturated flow and can be calibrated or inverted
against observations
([Šimůnek et al.,
2012](https://elibrary.asabe.org/abstract.asp?aid=42239)). Richards-equation
approaches provide a physically grounded way to model water movement through
soil, but they need hydraulic parameters, boundary conditions and often fine
soil/profile information
([Richards-equation discussion](https://www.sciencedirect.com/science/article/pii/S0022169402002512)).

Pros:

- physically interpretable;
- can represent depth, fluxes, drainage and root uptake;
- useful for scenario testing and ecophysiology coupling;
- can provide constraints for ML models.

Cons:

- parameter hungry;
- uncertain soil hydraulic properties at paddock scale;
- high computational cost for daily 30 m mapping;
- difficult to calibrate nationally without dense observations;
- may still need data assimilation or empirical correction.

### 5.5 Hybrid direction

The most promising future pathway may be hybrid:

- retain model6-style national covariate prediction;
- add public in-situ training data;
- include optical/SAR/thermal features where they improve shallow or surface
  prediction;
- enforce coarse-scale consistency with SMIPS;
- use local dense observations for calibration and uncertainty;
- test physics-informed constraints, e.g. water-balance limits or terrain-flow
  priors.

## 6. Public training data and future expansion

More public training data are a key future direction. Candidate sources include:

- ISMN stations where Australian and comparable international data are usable;
- CosmOz and cosmic-ray neutron probe products where accessible;
- OzNet / MSMMN extensions;
- OzFlux or flux-tower sites with soil-moisture probes;
- SMAP/SMOS calibration-validation networks;
- public state or catchment soil-moisture monitoring programs;
- citizen/agricultural networks if licensing and calibration are acceptable.

The main technical issue is harmonisation:

- depth intervals differ;
- sensor types differ;
- measurement support differs;
- time of day differs;
- calibration and QA/QC differ;
- root-zone and surface targets should not be mixed blindly.

Future training should probably maintain multiple targets:

1. surface / shallow soil moisture;
2. root-zone soil moisture;
3. local calibration residual;
4. uncertainty or confidence class.

## 7. Recommended next steps

1. **Clarify the dense-device target.** Confirm measurement depth and whether the
   dense dataset should train a shallow product, root-zone correction, or both.
2. **Rebuild the OzNet training table.** Then run true OzNet+local retraining
   rather than residual correction only.
3. **Formalise leave-property / leave-campaign validation.** Dense local
   retraining must be tested on another dense site or future campaign.
4. **Add public in-situ datasets.** Prioritise datasets with known depth,
   calibration and licensing.
5. **Create uncertainty outputs.** Map where the model is extrapolating in
   feature space.
6. **Add optional optical/SAR/thermal features.** Test whether they improve the
   shallow/dense-device product.
7. **Develop active-learning guidance.** Use the spiking curves to identify
   where the next sensor or point campaign would be most valuable.
8. **Add a coarse-consistency check.** Consider whether fine predictions should
   aggregate back toward SMIPS at the coarse-cell scale.

## 8. Human fill-in flags

- Add final project title, team, intended audience and funding context.
- Confirm whether the final product is intended for grazing management,
  restoration monitoring, tree water-use modelling, crop modelling or all of
  these.
- Confirm how the dense point device reports soil moisture and whether its
  values are directly comparable to volumetric water content.
- Decide how strongly to claim local retraining results, given the
  root-zone/shallow-device mismatch.
- Add any unpublished field context: vegetation, enclosure status, sampling
  design, rainfall timing and known problem points.
- Add preferred citation style and formal bibliography before external
  circulation.

## References and comparable studies

- Smith, A. B. et al. (2012). *The Murrumbidgee soil moisture monitoring network
  data set*. Water Resources Research.
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2012WR011976
- Peng, J. et al. (2017). *A review of spatial downscaling of satellite remotely
  sensed soil moisture*. Reviews of Geophysics.
  https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016RG000543
- Piles, M. et al. (2011). *Downscaling SMOS-derived soil moisture using MODIS
  visible/infrared data*. IEEE Transactions on Geoscience and Remote Sensing.
  https://www.documentation.ird.fr/hor/PAR00007912
- *Spatial Downscaling of Satellite-Based Soil Moisture Products Using Machine
  Learning Techniques: A Review* (2024). Remote Sensing.
  https://www.mdpi.com/2072-4292/16/12/2067
- *Downscaling satellite soil moisture using geomorphometry and machine learning*
  (2019). https://pmc.ncbi.nlm.nih.gov/articles/PMC6759172/
- *Downscaling Satellite Soil Moisture Using a Modular Spatial Inference
  Framework* (2022). Remote Sensing.
  https://www.mdpi.com/2072-4292/14/13/3137
- *Global downscaling of remotely sensed soil moisture using neural networks*
  (2018). HESS. https://hess.copernicus.org/articles/22/5341/2018/
- Beven, K. J. & Kirkby, M. J. (1979). *A physically based, variable
  contributing area model of basin hydrology*.
  https://hero.epa.gov/reference/3349401
- Šimůnek, J., van Genuchten, M. Th. & Šejna, M. (2012). *HYDRUS: Model use,
  calibration, and validation*. Transactions of the ASABE.
  https://elibrary.asabe.org/abstract.asp?aid=42239
