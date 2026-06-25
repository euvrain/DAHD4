import numpy as np


def center(x):
    """
    Center data by subtraction of the mean.

    If X contains NaNs indicating missing values, the mean is computed
    from the available data only.

    Args:
        x (np.ndarray): Input vector or 2-D array.

    Returns:
        tuple:
            xc (np.ndarray): Centered data with mean removed.
            xm (np.ndarray): Column-wise mean of x.

    Raises:
        ValueError: If x has more than 2 dimensions.

    See also:
        np.nanmean, np.mean
    """
    if x.ndim > 2:
        raise ValueError("X must be a vector or 2-D array.")

    # If x is a vector, make sure it is a column vector
    if len(x) == np.prod(x.shape):
        x = x.reshape(-1, 1)

    m, n = x.shape

    # Get mean of x
    if np.any(np.any(np.isnan(x))):     # there are missing values in x
        xm = np.nanmean(x, axis=0)
    else:                               # no missing values
        xm = np.mean(x, axis=0)

    # Remove mean
    xc = x - np.tile(xm, (m, 1))

    return xc, xm