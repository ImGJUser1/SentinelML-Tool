from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/utils/types.py
"""
Type definitions for SentinelML.
"""

from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import numpy.typing as npt

# Array types
ArrayLike = Union[
    npt.NDArray, Sequence[float], Sequence[Sequence[float]], "pd.DataFrame"  # Forward reference
]

PathLike = Union[str, Path]

# Model types
ModelLike = Union[
    "sklearn.base.BaseEstimator",  # sklearn models
    "torch.nn.Module",  # PyTorch models
    "tf.keras.Model",  # TensorFlow models
    Any,  # Other frameworks
]

# Prediction types
PredictionLike = Union[npt.NDArray, Sequence[float], Sequence[Sequence[float]]]

# Data types
DataFrameLike = Union["pd.DataFrame", "pl.DataFrame", Any]  # Polars
