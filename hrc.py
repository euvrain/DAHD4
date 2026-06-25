import numpy as np
from scipy.signal import lfilter


def hrc(A, E, L, index=None):
    """
    Calculate the Reconstructed Harmonic Components (RHCs).

    Uses the DAHD modes (E) and DAHD coefficients (A) corresponding to a
    given frequency band. Both matrices may have had columns trimmed but
    must have the same number of columns.

    Args:
        A (np.ndarray): DAHD coefficients (DAHC) matrix of shape (N-M'+1, k).
        E (np.ndarray): DAHD mode (DAHM) matrix of shape (L*M', k).
        L (int): Number of channels in the original data matrix.
        index (array-like, optional): Index vector into A and E. For example,
            if index = [1, 2], the HRC for combined modes 1 and 2 will be
            computed. Since DAHMs are paired, k should be an even number.
            Defaults to all columns.

    Returns:
        np.ndarray: Rs - Matrix of HRC for combination of input DAHMs/DAHCs,
            shape (N, L).

    Raises:
        ValueError: If E and A do not have the same number of columns.

    References:
        See Appendix A of Chekroun, M. D., and D. Kondrashov, 2017:
        Data-adaptive harmonic spectra and multilayer Stuart-Landau models,
        Chaos, 27, 093110: doi:10.1063/1.4989400

        Kondrashov D, Ryzhov EA, Berloff P. Data-adaptive harmonic analysis
        of oceanic waves and turbulent flows.
        Chaos, 30, doi: 10.1063/5.0012077

    Notes:
        Written by Dmitri Kondrashov. Version date 1/29/25.
        Please send comments and suggestions to dkondras@atmos.ucla.edu
    """
    ml, k = E.shape
    ra, ka = A.shape

    if index is None:
        index = np.arange(k)

    if k != ka:
        raise ValueError('E and A must have the same number of columns.')

    M = int(ml // L)    # These lines assume that E and A
    N = int(ra + M - 1) # have the "right" row dimensions.
    MT = ra

    R = np.zeros((N, L * len(index)))
    Z = np.zeros((M - 1, k))
    A = np.vstack([A, Z])   # Zero pad A to make it N by k
    Ej = np.zeros((M, L))

    # Calculate HRCs
    for j in range(len(index)):
        Ej = E[:, index[j]].reshape(M, L)  # Convert the j-th DAHM into a matrix of filters
        for i in range(L):                  # Compute the HRCs for the j-th DAHM/DAHC pair
            R[:, j * L + i] = lfilter(Ej[:, i], [M], A[:, index[j]])

    # Adjust first M-1 rows and last M-1 rows
    Rs = np.zeros((N, L))
    for j in range(len(index)):
        Rs = Rs + R[:, j * L:(j + 1) * L]

    # 1 <= idat < M
    for idat in range(1, MT + 1):       # MATLAB: 1:MT
        if idat < M:
            Rs[idat - 1, :] = Rs[idat - 1, :] * M / idat

    # N-M+1 < idat <= N
    for idat in range(MT + 1, N + 1):   # MATLAB: MT+1:N
        if idat <= M:
            # 1 <= i < M (short timeseries: N < 2M)
            Rs[idat - 1, :] = Rs[idat - 1, :] * M / MT
        else:
            # M <= i <= N-M+1
            Rs[idat - 1, :] = Rs[idat - 1, :] * M / (N - idat + 1)

    return Rs