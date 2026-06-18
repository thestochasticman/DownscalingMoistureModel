"""EMT: downscaling SMIPS soil moisture to topographic (~30 m) resolution.

EMT is a thin layer over PaddockTS: SMIPS and terrain are downloaded with
PaddockTS via its ``Query`` object (see ``emt.queries``), and EMT only adds
what PaddockTS lacks --

    insitu      -- OzNet Murrumbidgee in-situ root-zone soil moisture (ground truth)
    queries     -- PaddockTS Query builders for the EMT study areas
    covariates  -- terrain covariate stack + point sampling (uses PaddockTS)
    features, model, downscale, evaluate (to come)
"""

__version__ = "0.1.0"
