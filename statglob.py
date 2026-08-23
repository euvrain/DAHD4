"""
statglob.py

Diagnostic plots comparing a reference reconstruction ("data") against an
ensemble of MSLM simulations/forecasts ("datas"): lagged autocorrelation
(top panel) and probability density (bottom panel).

Python conversion of statglob.m (Dmitri Kondrashov) by Taylor McDonald, 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.stats import gaussian_kde


def _xcorr_coeff(x, maxlag):
    """
    Autocorrelation of a 1-D signal, normalized so the zero-lag value is 1
    -- matches MATLAB's `xcorr(x, maxlag, 'coeff')` for a single signal.

    Returns the lags 0..maxlag (MATLAB code only ever uses the non-negative
    half via `tmpcd(lead+1:end)`).
    """
    x = np.asarray(x).ravel()
    N = len(x)
    c = correlate(x, x, mode='full')  # length 2N-1, lags -(N-1)..(N-1)
    mid = N - 1
    c = c[mid - maxlag: mid + maxlag + 1]  # lags -maxlag..maxlag
    c = c / c[maxlag]  # zero-lag normalization ('coeff')
    return c


def _ksdensity(x, npoints=100):
    """
    Kernel density estimate approximating MATLAB's `ksdensity(x)` defaults:
    a Gaussian kernel evaluated at `npoints` points spanning the data range
    (padded by a few kernel bandwidths). Bandwidth selection differs
    slightly from MATLAB's (Scott's rule via `gaussian_kde` vs MATLAB's
    normal reference rule), so values may not match bit-for-bit, but the
    shape/behavior is equivalent.
    """
    x = np.asarray(x).ravel()
    x = x[~np.isnan(x)]
    kde = gaussian_kde(x)
    bw = kde.factor * np.std(x, ddof=1)
    pad = 4 * bw
    xi = np.linspace(x.min() - pad, x.max() + pad, npoints)
    f = kde(xi)
    return f, xi


def statglob(data, datas, lead):
    """
    Plot lagged-autocorrelation and PDF comparisons between a reference
    reconstruction and an ensemble of MSLM realizations.

    Args:
        data (np.ndarray): Reference reconstruction, shape (N0, L0).
        datas (np.ndarray): MSLM realizations, shape (N, K) with K a
            multiple of L0 -- reshaped internally to (N, L0, K/L0).
        lead (int): Maximum lag (in samples) for the autocorrelation panel.
    """
    data = np.asarray(data, dtype=float)
    datas = np.asarray(datas, dtype=float)

    N0, L0 = data.shape
    N, K = datas.shape
    n_ens = K // L0
    datas3 = datas.reshape(N, L0, n_ens)

    fig = plt.figure(figsize=(8, 10))

    # ── Top panel: lagged autocorrelation, reconstruction vs. MSLM ─────────
    ax1 = fig.add_subplot(2, 1, 1)
    for k in range(n_ens):
        cd = np.zeros((lead + 1, L0))
        c = np.zeros((lead + 1, L0))
        for i in range(L0):
            tmpd = data[:, i]
            cd[:, i] = _xcorr_coeff(tmpd - np.mean(tmpd), lead)[lead:]
            tmp = datas3[:, i, k]
            c[:, i] = _xcorr_coeff(tmp - np.mean(tmp), lead)[lead:]

        tmpc = np.mean(c, axis=1)
        tmpcd = np.mean(cd, axis=1)

        lags = np.arange(1, lead + 2)
        ax1.plot(lags, tmpcd, 'r', linewidth=2)
        ax1.plot(lags, tmpc, 'b', linewidth=2)

        ymin0 = np.min(cd)
        ymin1 = np.min(tmpc)
        ymin = min(ymin0, ymin1)
        ax1.set_ylim([ymin, 1])
        ax1.set_xlim([1, lead + 1])

    ax1.set_xlabel('Lag')
    ax1.set_title('Corr')
    ax1.legend(['Recons', 'MSLM'])
    ax1.tick_params(labelsize=16)

    # ── Bottom panel: PDF, reconstruction vs. every MSLM realization ───────
    ax2 = fig.add_subplot(2, 1, 2)

    datas_flat = datas3.reshape(N * L0, n_ens)
    data_flat = data.reshape(N0 * L0)

    f, xi = _ksdensity(data_flat)
    ax2.semilogy(xi, f, 'r', linewidth=2, label='Recons')

    for k in range(n_ens):
        f, xi = _ksdensity(datas_flat[:, k])
        ax2.semilogy(xi, f, 'b', linewidth=1, label='MSLM' if k == 0 else None)

    ax2.legend(['Recons', 'MSLM'])
    ax2.set_xlim([-8, 8])
    ax2.set_title('PDF')
    ax2.tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()
