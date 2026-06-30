from scipy.signal import correlate
import numpy as np

def xcorr(x, y, maxlag, method='unbiased'):
    """
    Cross-correlate two 1-D signals, matching MATLAB's xcorr behavior.

    Args:
        x (np.ndarray): First input signal.
        y (np.ndarray): Second input signal.
        maxlag (int): Maximum lag to compute (returns lags -maxlag..maxlag).
        method (str): Normalization method. 'unbiased' divides each lag by
            the number of overlapping samples (N - |lag|). 'biased' divides
            by N. 'none' applies no normalization.

    Returns:
        np.ndarray: Cross-correlation values of length 2*maxlag + 1.
    """
    N = len(x)
    c = correlate(x, y, mode='full')  # length 2N - 1, lags -(N-1)..(N-1)

    mid = N - 1
    c = c[mid - maxlag: mid + maxlag + 1]  # trim to -maxlag..maxlag

    if method == 'unbiased':
        lags = np.arange(-maxlag, maxlag + 1)
        norm = N - np.abs(lags)
        c = c / norm
    elif method == 'biased':
        c = c / N

    return c