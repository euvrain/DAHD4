import numpy as np


def DAHM4_ex(vv, W, NF, NT):
    """
    Compute EVP - space-time DAHMs via inverse Fourier transform.

    Computes space-time Data-Adaptive Harmonic Modes (DAHMs) obtained
    by an inverse Fourier transform of Hermitian DAHD eigenvectors vv.

    Args:
        vv (np.ndarray): Hermitian DAHD eigenvectors.
        W (int): Embedding window.
        NF (int): Frequency index.
        NT (int): Number of DAHMs to compute.

    Returns:
        np.ndarray: EVP - space-time DAHMs matrix.

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
    D = vv.shape[0]

    EVP = np.zeros(((2 * W - 1) * D, NT))

    fftv = np.zeros((2 * W - 1, D, NT), dtype=complex)

    if NF > 1:
        for i in range(NT // 2):
            i2 = 2 * i
            cvv = 1j * vv[:, i]
            fftv[NF, :, i2] = np.sqrt(W - 0.5) * vv[:, i]
            fftv[-NF, :, i2] = np.sqrt(W - 0.5) * np.conj(vv[:, i])
            fftv[NF, :, i2 + 1] = np.sqrt(W - 0.5) * cvv
            fftv[-NF, :, i2 + 1] = np.sqrt(W - 0.5) * np.conj(cvv)
    else:
        for i in range(NT // 2):
            cvv = 1j * vv[:, i]
            fftv[NF, :, i] = np.sqrt(2 * W - 1) * vv[:, i]

    Evv = np.fft.ifft(fftv, axis=0)
    EVP = np.reshape(Evv, (Evv.shape[0] * Evv.shape[1], Evv.shape[2]))

    return EVP