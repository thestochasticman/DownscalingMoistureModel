"""EMT: downscaling SMIPS soil moisture to topographic (~30 m) resolution.

Pipeline stages:
    1. insitu  -- OzNet Murrumbidgee in-situ root-zone soil moisture (ground truth)
    2. smips   -- coarse (~1 km) soil moisture cube to be downscaled
    3. terrain -- 30 m DEM + derivatives (the target grid / fine covariates)
    4. features, 5. model, 6. downscale, 7. evaluate
"""

__version__ = "0.1.0"
