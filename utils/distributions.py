import time
from typing import Optional

import numpy as np

# Error messages
INVALID_PARAMS_MSG = (
    "gauss: Enter a valid pair of either the lower and upper bound or the mean and the standard deviation."
)
INVALID_BOUNDS_MSG = (
    "gauss: The lower-bound of the Gaussian distribution should be strictly smaller than the upper-bound."
)
INVALID_STD_DEV_MSG = "gauss: Standard deviation should be a positive number"
MISSING_BOUNDS_MSG = "gauss: Both a and b must be provided when mean or std_dev are not specified."
TRIRND_INVALID_BOUNDS_MSG = "trirnd: The upper bound must be strictly greater than the lower bound"
TRIRND_INVALID_PEAK_MSG = "trirnd: The peak must be between the lower and upper bound, inclusive."


def gauss(m: int, n: int, a: Optional[float] = None, b: Optional[float] = None, mean=None, std_dev=None):
    """
    Generate a random array of size m by n sampled from a Gaussian distribution.

    Parameters:
    - a: The lower-bound cutoff for the distribution sampling
    - b: The upper-bound cutoff for the distribution sampling
    - m: The number of rows in the output array.
    - n: The number of columns in the output array.

    Parameters (optional):
    - mean: the mean of the distribution
    - std_dev: the standard deviation of the distribution

    Returns:
    - An array of shape (m, n) containing random samples from the Gaussian distribution.

    Authors: Priyadarshan (priyada@purdue.edu), Alex Lee (alexlee5124@gmail.com)
    Date: 12/12/2024
    """
    if (a is None or b is None) and (mean is None or std_dev is None):
        raise ValueError(INVALID_PARAMS_MSG)
    if (a is not None and b is not None) and (a >= b):
        raise ValueError(INVALID_BOUNDS_MSG)
    if std_dev is not None and std_dev <= 0:
        raise ValueError(INVALID_STD_DEV_MSG)

    if mean is None or std_dev is None:
        if a is None or b is None:
            raise ValueError(MISSING_BOUNDS_MSG)
        mean = (a + b) / 2.0
        std_dev = (b - mean) / 2.58

    # Generate a random sample from the Gaussian distribution with specified mean and std_dev
    np.random.seed(seed=int(time.time()))
    return np.random.normal(loc=mean, scale=std_dev, size=(m, n)).tolist()


def trirnd(a: float, c: float, m: int, n: int, mode=None):
    """
    Generate a random array of size m by n sampled from a Gaussian distribution.

    Parameters:
    - a: The lower-bound cutoff for the distribution sampling
    - c: The upper-bound cutoff for the distribution sampling
    - m: The number of rows in the output array.
    - n: The number of columns in the output array.

    Parameters (optional):
    - mode: the peak of the triangular distribution

    Returns:
    - An m x n matrix with values sampled from the triangular distribution.

    Authors: Priyadarshan (priyada@purdue.edu), Alex Lee (alexlee5124@gmail.com)
    Date: 12/12/2024
    """
    if a >= c:
        raise ValueError(TRIRND_INVALID_BOUNDS_MSG)

    if mode is None:
        mode = (a + c) / 2.0
    if (mode < a) or (mode > c):
        raise ValueError(TRIRND_INVALID_PEAK_MSG)

    # Generate a random sample from a triangular distribution with specified lower bound, mode, and upper bound
    np.random.seed(seed=int(time.time()))
    return np.random.triangular(a, mode, c, size=(m, n)).tolist()
