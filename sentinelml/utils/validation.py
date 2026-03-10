from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
# sentinelml/utils/validation.py
"""
Input validation utilities.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


def validate_array(
    X: Any,
    accept_sparse: bool = False,
    dtype: Optional[str] = None,
    force_2d: bool = True,
    allow_nd: bool = False,
) -> npt.NDArray:
    """
    Validate and convert input to numpy array.

    Parameters
    ----------
    X : array-like
        Input data.
    accept_sparse : bool
        Whether to accept sparse matrices.
    dtype : str, optional
        Target data type.
    force_2d : bool
        Whether to force 2D array.
    allow_nd : bool
        Whether to allow >2D arrays.

    Returns
    -------
    X : ndarray
        Validated array.
    """
    from sklearn.utils.validation import check_array as sklearn_check_array

    return sklearn_check_array(
        X,
        accept_sparse=accept_sparse,
        dtype=dtype,
        force_all_finite="allow-nan",
        ensure_2d=force_2d,
        allow_nd=allow_nd,
    )


def validate_inputs(
    X: Any, y: Optional[Any] = None, multi_output: bool = False
) -> Tuple[npt.NDArray, Optional[npt.NDArray]]:
    """
    Validate X and y inputs.

    Parameters
    ----------
    X : array-like
        Features.
    y : array-like, optional
        Targets.
    multi_output : bool
        Whether multiple outputs are allowed.

    Returns
    -------
    X, y : validated arrays
    """
    from sklearn.utils.validation import check_X_y

    if y is not None:
        X, y = check_X_y(X, y, multi_output=multi_output, force_all_finite="allow-nan")
        return X, y

    X = validate_array(X)
    return X, None


def check_non_negative(X: npt.NDArray, whom: str = "Input") -> npt.NDArray:
    """Ensure array contains no negative values."""
    X = np.asarray(X)
    if np.any(X < 0):
        raise ValueError(f"{whom} must contain only non-negative values")
    return X


def check_finite(X: npt.NDArray, whom: str = "Input") -> npt.NDArray:
    """Ensure array contains no NaN or Inf values."""
    X = np.asarray(X)
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{whom} contains NaN or infinite values")
    return X
