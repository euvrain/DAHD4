import numpy as np
import scipy.signal as signal
from scipy.linalg import eigh

from xcorr import xcorr


def DAHD4freq_part_weight(X, W, NFE, NP, wt):
    """
    Calculate the DAHD spectrum in frequency-domain formulation.

    Args:
        X (np.ndarray): Input data array of shape (N, L).
        W (int): Embedding window M.
        NFE (int): Number of frequencies (maximum is M).
        NP (int): Number of spectral elements per frequency (maximum is L).
        wt (str): Method for weighting temporal correlations. One of
            'hamming', 'hanning', 'blackman', 'hann', 'bartlett', or any
            other value for no weighting (uniform window).

    Returns:
        tuple:
            fE2 (np.ndarray): Resolved frequency bins, size (M,).
            VP (np.ndarray): Hermitian DAHD eigenvalues sorted by frequency,
                max size (L, M).
            FEP (np.ndarray): Hermitian DAHD eigenvectors sorted by
                frequency, max size (L, L, M).

    References:
        Kondrashov D. et al. 2020 Data-adaptive harmonic analysis of
        oceanic waves and turbulent flows.
        Chaos, 30, doi: 10.1063/5.0012077

        Kondrashov D. et al. 2026 Accurate and robust real-time prediction
        of September Arctic sea ice.
        Chaos: 36, doi: 10.1063/5.0295634

    Notes:
        Written by Dmitri Kondrashov. Version date 6/22/26.
        Please send comments and suggestions to dkondras@atmos.ucla.edu
    """
    WW = 2 * W - 1
    dim = X.shape[1]
    cspec = np.zeros((WW, dim, dim))

    # Select temporal weighting window
    if wt == 'hamming':
        weight = signal.windows.hamming(WW, sym=True)
    elif wt == 'hanning':
        weight = signal.windows.hann(WW, sym=True)
    elif wt == 'blackman':
        weight = signal.windows.blackman(WW, sym=True)
    elif wt == 'hann':
        weight = signal.windows.hann(WW, sym=True)
    elif wt == 'bartlett':
        weight = signal.windows.bartlett(WW)
    else:
        weight = np.ones(WW)

    # Weighted cross-correlation spectrum
    method = 'unbiased'
    for k in range(dim):
        for l in range(dim):
            cspec[:, k, l] = weight * xcorr(X[:, l], X[:, k], W - 1, method)

    toto = np.fft.fft(cspec, axis=0)

    VP = np.zeros((NP, NFE))
    VPT = np.zeros((NP, NFE))
    FEP = np.zeros((dim, NP, NFE))

    for NF in range(NFE):
        cf2 = np.exp(-1j * NF * np.pi / WW)
        toto2 = np.squeeze(toto[NF, :, :] * cf2)
        if np.trace(toto2) < 0:
            toto2 = -toto2

        # Top NP eigenvalues/eigenvectors (eigh sorts ascending, so reverse)
        eigenvalues, ee = eigh(toto2, subset_by_index=[dim - NP, dim - 1])
        eigenvalues = eigenvalues[::-1]
        ee = ee[:, ::-1]

        VPT[:, NF] = np.real(eigenvalues)
        VP[:, NF] = np.abs(eigenvalues)
        FEP[:, :, NF] = np.conj(ee)

        numPositive = np.sum(eigenvalues > 0)
        numNegative = np.sum(eigenvalues < 0)
        print(f"{NF} >0:{numPositive} <0:{numNegative}")

    # Resolved frequency bins
    M1 = 2 * W - 1
    MM2 = (M1 - 1) / 2
    fE2 = np.zeros(NFE)
    for j in range(NFE):
        fE2[j] = 0.5 * j / MM2

    return fE2, VP, FEP