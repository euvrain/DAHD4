import numpy as np
from scipy.linalg import lstsq


def _center(x):
    """Subtract column-wise mean. Returns (centered_x, mean)."""
    xm = np.mean(x, axis=0)
    return x - xm, xm


def MSLM_FCST(LEAD, data, NELIN, NLEVEL, NSMT, ires, inorm, iSYM, iNL, inEQ, indrandperm):
    """
    Perform forecast using multi-level stochastic Stuart-Landau modeling
    enabled by DAHD analysis.

    Args:
        LEAD (int): Maximum time steps to forecast.
        data (np.ndarray): Array of DAHCs from DAHD analysis, shape (N, nmax).
        NELIN (int): Model specification — 1 for Stuart-Landau, 0 for linear.
        NLEVEL (int): Number of model levels.
        NSMT (int): Number of stochastic realizations.
        ires (int): Noise specification — 0 for correlated white noise,
            1 for permutation of last-level regression residuals.
        inorm (int): Data normalization flag — 0 or 1.
        iSYM (int): Symmetry constraint flag for SL model.
        iNL (int): Nonlinear equality constraint flag for SL model.
        inEQ (int): Inequality constraint flag for SL model.
        indrandperm (np.ndarray): Array of random indices for stochastic ensemble.

    Returns:
        tuple:
            XSM (np.ndarray): Predicted data, shape (N+LEAD, nmax, NSMT).
            xt_res (np.ndarray): Regression residuals at last model level.
            L (np.ndarray): Array of model coefficients.
            ENL (np.ndarray): Array of nonlinear model coefficients.

    References:
        Kondrashov D. et al. 2020 Data-adaptive harmonic analysis of
        oceanic waves and turbulent flows.
        Chaos, 30, doi: 10.1063/5.0012077

        Kondrashov D. et al. 2026 Accurate and robust real-time prediction
        of September Arctic sea ice.
        Chaos: 36, doi: 10.1063/5.0295634

    Notes:
        Written by Dmitri Kondrashov. Version date 7/8/26.
        Python conversion by Taylor McDonald, SETI Institute, 2026.
        Please send comments and suggestions to dkondras@atmos.ucla.edu
    """

    # ── Initialization ────────────────────────────────────────────────────────
    meandata = np.mean(data, axis=0)
    DMD, _   = _center(data)

    nmax = DMD.shape[1]
    NTT  = DMD.shape[0]

    XT_RES = np.zeros((NTT - 1, nmax, NLEVEL))
    XX     = np.zeros((NTT,     nmax, NLEVEL))
    L      = np.zeros((nmax, nmax * NLEVEL, NLEVEL))
    ENL    = np.zeros((nmax, 1))
    F      = np.zeros((nmax, NLEVEL))
    stdr   = np.zeros((nmax, NLEVEL))

    XX[:NTT, :, 0] = DMD
    stddata = np.std(XX[:NTT, :, 0], axis=0)
    stddata = np.where(stddata == 0, 1.0, stddata)  # avoid division by zero

    if inorm == 1:
        XX[:NTT, :, 0] = XX[:NTT, :, 0] / stddata

    # ── Model fitting loop over levels ────────────────────────────────────────
    for nl in range(NLEVEL):

        xt  = np.diff(XX[:NTT, :, nl], axis=0)
        NTT = xt.shape[0]

        if nl == 0:
            NTE = NTT

            if NELIN == 1:
                for n0 in range(nmax // 2):

                    gsvd = np.zeros((NTE * 2, (nmax + 2) * 2))
                    xg   = np.zeros(NTE * 2)
                    indk = [2 * n0, 2 * n0 + 1]

                    for n_1 in range(2):
                        n = indk[n_1]
                        nl_term = (XX[:NTE, n, 0] *
                                   (XX[:NTE, indk[0], 0] ** 2 +
                                    XX[:NTE, indk[1], 0] ** 2))
                        pred = np.column_stack([
                            np.ones(NTE),
                            XX[:NTE, :, 0],
                            nl_term
                        ])
                        r0 = n_1 * NTE
                        c0 = n_1 * (nmax + 2)
                        gsvd[r0:r0 + NTE, c0:c0 + (nmax + 2)] = pred
                        xc, _ = _center(xt[:, n:n + 1])
                        xg[r0:r0 + NTE] = xc[:, 0]

                    # FIX: scipy.linalg.lstsq instead of lsq_linear
                    bg, _, _, _ = lstsq(gsvd, xg, check_finite=False)
                    residual = xg - gsvd @ bg

                    for n_1 in range(2):
                        n  = indk[n_1]
                        c0 = n_1 * (nmax + 2)
                        L[n, :nmax, 0] = bg[c0 + 1: c0 + nmax + 1]
                        ENL[n, 0]      = bg[c0 + nmax + 1]
                        F[n, 0]        = bg[c0]

                    XT_RES[:NTT, indk[0]:indk[1] + 1, 0] = residual.reshape(NTT, 2)

            if NELIN == 0:
                for n_1 in range(nmax):
                    A = np.column_stack([np.ones(NTE), XX[:NTE, :, 0]])
                    b = xt[:NTE, n_1]
                    beta, _, _, _ = lstsq(A, b, check_finite=False)
                    L[n_1, :nmax, 0]     = beta[1:nmax + 1]
                    F[n_1, 0]            = beta[0]
                    XT_RES[:NTE, n_1, 0] = b - A @ beta
                    ENL[n_1]             = 0

        else:
            if NELIN == 1:
                for n0 in range(nmax // 2):
                    indk = [2 * n0, 2 * n0 + 1]
                    inds = list(indk)
                    for nl_1 in range(nl):
                        inds += [i + (nl_1 + 1) * nmax for i in indk]

                    pred_list = []
                    for nl_1 in range(nl):
                        pred_list.append(np.squeeze(XX[:NTT, indk, nl_1]))
                    pred = np.hstack(pred_list) if pred_list else np.zeros((NTT, 0))

                    for n in range(indk[0], indk[1] + 1):
                        xc, _ = _center(xt[:, n:n + 1])
                        if pred.shape[1] > 0:
                            beta, _, _, _ = lstsq(pred, xc[:, 0], check_finite=False)
                            L[n, inds[:len(beta)], nl]  = beta
                            XT_RES[:NTT, n, nl] = xc[:, 0] - pred @ beta
                        else:
                            XT_RES[:NTT, n, nl] = xc[:, 0]

            else:
                pred_list = []
                for nl_1 in range(nl):
                    pred_list.append(np.squeeze(XX[:NTT, :, nl_1]))
                pred = np.hstack(pred_list) if pred_list else np.zeros((NTT, 0))
                A = np.column_stack([np.ones(NTT), pred])

                for n in range(nmax):
                    xc, _ = _center(xt[:, n:n + 1])
                    beta, _, _, _ = lstsq(A, xc[:, 0], check_finite=False)
                    L[n, :nmax * nl, nl] = beta[1:]
                    F[n, nl]             = beta[0]
                    XT_RES[:NTT, n, nl]  = xc[:, 0] - A @ beta

        # avoid division by zero in stdr
        stdr[:, nl] = np.std(XT_RES[:NTT, :nmax, nl], axis=0)
        stdr[:, nl] = np.where(stdr[:, nl] == 0, 1.0, stdr[:, nl])

        if nl != NLEVEL - 1:
            xc, _ = _center(XT_RES[:NTT, :nmax, nl])
            XX[:NTT, :nmax, nl + 1] = xc
            if inorm == 1:
                XX[:NTT, :nmax, nl + 1] = XX[:NTT, :nmax, nl + 1] / stdr[:, nl]

    # ── Build full system matrix ──────────────────────────────────────────────
    matl = np.zeros((nmax * NLEVEL, nmax * NLEVEL))
    for nl in range(NLEVEL):
        for n in range(nmax):
            matl[n + nl * nmax, :nmax * (nl + 1)] = L[n, :nmax * (nl + 1), nl]
    for nl in range(NLEVEL - 1):
        for n in range(nmax):
            cof = stdr[n, nl] if inorm == 1 else 1.0
            matl[n + nl * nmax, n + (nl + 1) * nmax] = cof

    EL, EV = np.linalg.eig(matl)
    indel  = np.where(np.real(EL) > 0)[0]
    ind    = np.where(ENL.ravel() > 0)[0]

    # ── Stochastic forecast setup ─────────────────────────────────────────────
    xt_res = XT_RES[:NTT, :, NLEVEL - 1]

    # Robust covariance: handle NaN, symmetrize, jitter until Cholesky succeeds
    covn = np.nan_to_num(
        np.corrcoef(XT_RES[:NTT, :, NLEVEL - 1].T),
        nan=0.0, posinf=1.0, neginf=0.0
    )
    covn = (covn + covn.T) / 2   # enforce symmetry

    rr     = None
    jitter = 1e-8
    for _ in range(15):
        try:
            rr = np.linalg.cholesky(covn)
            break
        except np.linalg.LinAlgError:
            covn += np.eye(covn.shape[0]) * jitter
            jitter *= 10
    if rr is None:
        rr = np.eye(covn.shape[0])   # last resort: identity

    NE  = DMD.shape[0]
    NEE = NE + LEAD
    XSM = np.zeros((NEE, nmax, NSMT))

    iter_count = 0
    NSM        = 0

    # ── Forecast integration loop ─────────────────────────────────────────────
    while NSM < NSMT:
        iter_count += 1

        for NT in range(NE - NLEVEL, NEE - 1):
            if np.isnan(XX[NT, 0, 0]):
                break

            for nl in range(NLEVEL):
                pred1 = np.hstack([
                    np.squeeze(XX[NT, :, nl_1])
                    for nl_1 in range(nl + 1)
                ])
                tmp = L[:, :nmax * (nl + 1), nl] @ pred1

                if nl < NLEVEL - 1:
                    cof = stdr[:, nl] if inorm == 1 else np.ones(nmax)
                    nr  = XX[NT, :, nl + 1] * cof
                    XX[NT + 1, :, nl] = F[:, nl] + XX[NT, :, nl] + tmp + nr
                else:
                    if ires == 0:
                        rn = rr @ indrandperm[:nmax, NT, iter_count - 1] * stdr[:, NLEVEL - 1]
                    else:
                        ttt    = NT - NE + NLEVEL
                        timest = int(indrandperm[ttt, iter_count - 1])
                        rn     = XT_RES[timest, :, -1]
                    XX[NT + 1, :, nl] = F[:, nl] + XX[NT, :, nl] + tmp + rn

                if nl == 0 and NELIN == 1:
                    for n0 in range(nmax // 2):
                        indk = [2 * n0, 2 * n0 + 1]
                        for n in indk:
                            pred_nl = (XX[NT, n, 0] *
                                       (XX[NT, indk[0], 0] ** 2 +
                                        XX[NT, indk[1], 0] ** 2))
                            XX[NT + 1, n, 0] += ENL[n, 0] * pred_nl

        if not np.any(np.isnan(XX[:, :, 0])):
            NSM += 1
            if inorm == 1:
                tmp1 = np.squeeze(XX[:NEE, :, 0])
                XSM[:NEE, :, NSM - 1] = tmp1 * stddata + meandata
            else:
                XSM[:NEE, :, NSM - 1] = XX[:NEE, :, 0] + meandata

    return XSM, xt_res, L, ENL