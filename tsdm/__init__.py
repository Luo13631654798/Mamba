r"""Time Series Datasets and Models (TSDM).

Provides
  1. Facility to import some commonly used time series dataset
  2. Facility to import some commonly used time series models
  3. Facility to preprocess time series dataset

More complicated examples:

Random Search / Grid Search Hyperparameter optimization with nested cross-validation
split on a slurm cluster.

General idea:

1. Datasets should store data in "original" / "clean" / "pure form"
    - all kinds of data types allowed
    - all data types must support NaN values (-> pandas Int64 and StringDType !)
2. DataLoaders perform 2 tasks
    1. Encoding the data into pure float tensors
        - Consider different kinds of encoding
    2. Creating generator objects
        - random sampling from dataset
        - batching of random samples
        - caching?
"""

__all__ = [
    # Objects
    # Constants
    # "__version__",
    # Sub-Modules
    "config",
    "datasets",
    "encoders",
    "logutils",
    "metrics",
    "models",
    "optimizers",
    "viz",
    "random",
    "tasks",
    "utils",
]

import sys
from importlib import metadata

# version check
# if sys.version_info < (3, 10):
#     raise RuntimeError("Python >= 3.10 required")

# pylint: disable=wrong-import-position

from tsdm import (
    config,
    datasets,
    encoders,
    logutils,
    metrics,
    models,
    optimizers,
    random,
    tasks,
    utils,
    viz,
)