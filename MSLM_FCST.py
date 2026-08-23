import numpy as np
from scipy.linalg import lstsq
from center import center
from lsqlin_util import lsqlin


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
        Python conversion by Taylor McDonald, SETI Institute, and Dmitri Kondrashov 2026.
        Please send comments and suggestions to dkondras@atmos.ucla.edu
    """

    # ── Initialization ────────────────────────────────────────────────────────
    meandata = np.mean(data, axis=0)
    DMD, _   = _center(data)

    nmax = DMD.shape[1]
    NTT  = DMD.shape[0]
    NXX  = NTT + max(LEAD, 0) + 1

    XT_RES = np.zeros((NTT - 1, nmax, NLEVEL))
    XX     = np.zeros((NXX,     nmax, NLEVEL))
    L      = np.zeros((nmax, nmax * NLEVEL, NLEVEL))
    ENL    = np.zeros((nmax, 1))
    F      = np.zeros((nmax, NLEVEL))
    stdr   = np.zeros((nmax, NLEVEL))

    XX[:NTT, :, 0] = DMD
    stddata = np.std(XX[:NTT, :, 0], axis=0)
    stddata = np.where(stddata == 0, 1.0, stddata)

    if inorm == 1:
        XX[:NTT, :, 0] = XX[:NTT, :, 0] / stddata

    block = nmax + 2  # per-oscillator block layout: [intercept, nmax linear coeffs, nonlinear coeff]

    # ── Fit one regression model per level ──────────────────────────────────
    for nl in range(NLEVEL):
        xt = np.diff(XX[:NTT, :, nl], axis=0)
        NTT = xt.shape[0]

        if nl == 0:
            NTE = xt.shape[0]

            if NELIN == 1:
                # Stuart-Landau pair model, fit with true equality/inequality
                # constrained least squares (mirrors MATLAB's lsqlin call).
                for n0 in range(nmax // 2):
                    indk = [2 * n0, 2 * n0 + 1]

                    gsvd = np.zeros((NTE * 2, block * 2))
                    xg = np.zeros(NTE * 2)

                    for n_1 in range(2):
                        n = indk[n_1]
                        nl_term = XX[:NTE, n, 0] * (
                            XX[:NTE, indk[0], 0] ** 2 + XX[:NTE, indk[1], 0] ** 2
                        )
                        pred = np.column_stack([np.ones(NTE), XX[:NTE, :, 0], nl_term])

                        r0 = n_1 * NTE
                        c0 = n_1 * block
                        gsvd[r0:r0 + NTE, c0:c0 + block] = pred

                        xc, _ = center(xt[:, n:n + 1])
                        xg[r0:r0 + NTE] = xc[:, 0]

                    # ---- constraints, mirroring MATLAB's Aeq/beq/Aneq/bneq ----
                    nSYM = 2 if iSYM == 1 else 0
                    nNL = 1 if iNL == 1 else 0

                    Aeq = np.zeros((nSYM + nNL, block * 2))
                    beq = np.zeros(Aeq.shape[0])

                    if inEQ == 1:
                        Aneq = np.zeros((2, block * 2))
                        bneq = np.zeros(Aneq.shape[0])
                    else:
                        Aneq = None
                        bneq = None

                    # Column layout within a block (0-indexed): 0 = intercept,
                    # 1..nmax = linear coeffs, block-1 (== nmax+1) = nonlinear coeff.
                    diag1 = 0 * block + (indk[0] + 1)
                    off1 = 0 * block + (indk[1] + 1)
                    diag2 = 1 * block + (indk[1] + 1)
                    off2 = 1 * block + (indk[0] + 1)
                    nl_col_blk1 = 0 * block + (block - 1)
                    nl_col_blk2 = 1 * block + (block - 1)

                    if iSYM == 1:
                        # Diagonal (self-)coefficients equal between the pair.
                        Aeq[0, diag1] = 1
                        Aeq[0, diag2] = -1
                        # Off-diagonal (cross-)coefficients anti-symmetric.
                        Aeq[1, off1] = 1
                        Aeq[1, off2] = 1

                    if iNL == 1:
                        # Nonlinear coefficients equal between the pair.
                        Aeq[nSYM, nl_col_blk1] = 1
                        Aeq[nSYM, nl_col_blk2] = -1

                    if inEQ == 1:
                        # Nonlinear coefficients constrained non-positive (damping).
                        Aneq[0, nl_col_blk1] = 1
                        Aneq[1, nl_col_blk2] = 1

                    b0 = np.ones(block * 2)
                    bg, _ = lsqlin(gsvd, xg, Aeq=Aeq, beq=beq, Aneq=Aneq, bneq=bneq, x0=b0)
                    # print(f"DAHC Pair: {n0 + 1}, condition # {np.linalg.cond(gsvd):.6g}")

                    for n_1 in range(2):
                        n = indk[n_1]
                        c0 = n_1 * block
                        L[n, :nmax, 0] = bg[c0 + 1: c0 + nmax + 1]
                        ENL[n, 0] = bg[c0 + block - 1]
                        F[n, 0] = bg[c0]

                    residual = xg - gsvd @ bg
                    #XT_RES[:NTT, indk[0]:indk[1] + 1, 0] = residual.reshape(NTT, 2)
                    XT_RES[:NTT, indk[0]:indk[1] + 1, 0] = residual.reshape(NTT, 2, order='F')

            if NELIN == 0:
                for n_1 in range(nmax):
                    A = np.column_stack([np.ones(NTE), XX[:NTE, :, 0]])
                    b = xt[:NTE, n_1]
                    beta, _, _, _ = lstsq(A, b, check_finite=False)
                    L[n_1, :nmax, 0] = beta[1:nmax + 1]
                    F[n_1, 0] = beta[0]
                    XT_RES[:NTE, n_1, 0] = b - A @ beta
                    ENL[n_1, 0] = 0

        else:
            if NELIN == 1:
                for n0 in range(nmax // 2):
                    indk = [2 * n0, 2 * n0 + 1]

                    # Predictors: the pair's own two components at every level
                    # 0..nl (inclusive of the current level).
                    inds = []
                    for lvl in range(nl + 1):
                        inds += [i + lvl * nmax for i in indk]

                    pred = np.hstack([np.squeeze(XX[:NTT, indk, lvl]) for lvl in range(nl + 1)])

                    for n in range(indk[0], indk[1] + 1):
                        xc, _ = center(xt[:, n:n + 1])
                        beta, _, _, _ = lstsq(pred, xc[:, 0], check_finite=False)
                        L[n, inds, nl] = beta
                        XT_RES[:NTT, n, nl] = xc[:, 0] - pred @ beta

            else:
                # Predictors: the full state at every level 0..nl (inclusive).
                pred = np.hstack([np.squeeze(XX[:NTT, :, lvl]) for lvl in range(nl + 1)])
                A = np.column_stack([np.ones(NTT), pred])
                for n in range(nmax):
                    xc, _ = center(xt[:, n:n + 1])
                    beta, _, _, _ = lstsq(A, xc[:, 0], check_finite=False)
                    L[n, :nmax * (nl + 1), nl] = beta[1:]
                    F[n, nl] = beta[0]
                    XT_RES[:NTT, n, nl] = xc[:, 0] - A @ beta

        stdr[:, nl] = np.std(XT_RES[:NTT, :nmax, nl], axis=0)
        stdr[:, nl] = np.where(stdr[:, nl] == 0, 1.0, stdr[:, nl])

        if nl != NLEVEL - 1:
            xc, _ = center(XT_RES[:NTT, :nmax, nl])
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

    covn = np.nan_to_num(
        np.corrcoef(XT_RES[:NTT, :, NLEVEL - 1].T),
        nan=0.0, posinf=1.0, neginf=0.0
    )
    covn = (covn + covn.T) / 2

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
        rr = np.eye(covn.shape[0])

    NE  = DMD.shape[0]
    NEE = NE + LEAD
    XSM = np.zeros((NEE, nmax, NSMT))

    # Clip threshold to prevent runaway values
    CLIP_VAL = 1e3

    iter_count = 0
    NSM        = 0

    # ── Forecast integration loop ─────────────────────────────────────────────
    while NSM < NSMT:
        iter_count += 1

        # Reset XX forecast portion for each realization
        XX[NE:, :, :] = 0.0

        for NT in range(NE - NLEVEL, NEE - 1):
            if not np.all(np.isfinite(XX[NT, :, 0])):
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
                    val = F[:, nl] + XX[NT, :, nl] + tmp + nr
                else:
                    ic = (iter_count - 1) % indrandperm.shape[2]
                    if ires == 0:
                        rn = rr @ indrandperm[:nmax, NT % indrandperm.shape[1], ic] * stdr[:, NLEVEL - 1]
                    else:
                        ttt    = NT - NE + NLEVEL
                        timest = int(indrandperm[ttt % indrandperm.shape[0], ic])
                        rn     = XT_RES[timest % NTT, :, -1]
                    val = F[:, nl] + XX[NT, :, nl] + tmp + rn

                XX[NT + 1, :, nl] = np.clip(val, -CLIP_VAL, CLIP_VAL)

                # Stuart-Landau nonlinear correction at level 0
                # ENL is forced <= 0 during fitting so this term damps, not amplifies
                if nl == 0 and NELIN == 1:
                    for n0 in range(nmax // 2):
                        indk = [2 * n0, 2 * n0 + 1]
                        for n in indk:
                            x0 = XX[NT, n, 0]
                            x1 = XX[NT, indk[0], 0]
                            x2 = XX[NT, indk[1], 0]
                            if np.isfinite(x0) and np.isfinite(x1) and np.isfinite(x2):
                                pred_nl = x0 * (x1 ** 2 + x2 ** 2)
                                nl_term = ENL[n, 0] * pred_nl
                                if np.isfinite(nl_term):
                                    XX[NT + 1, n, 0] = np.clip(
                                        XX[NT + 1, n, 0] + nl_term,
                                        -CLIP_VAL, CLIP_VAL
                                    )

        if np.all(np.isfinite(XX[:NEE, :, 0])):
            NSM += 1
            if inorm == 1:
                tmp1 = np.squeeze(XX[:NEE, :, 0])
                XSM[:NEE, :, NSM - 1] = tmp1 * stddata + meandata
            else:
                XSM[:NEE, :, NSM - 1] = XX[:NEE, :, 0] + meandata

        # Safety: avoid infinite loop
        if iter_count > NSMT * 10:
            print(f"Warning: only {NSM}/{NSMT} valid realizations after {iter_count} attempts")
            # Fill remaining with last valid or zeros
            if NSM > 0:
                for k in range(NSM, NSMT):
                    XSM[:, :, k] = XSM[:, :, NSM - 1]
            break

    return XSM, xt_res, L, ENL