"""
MSLM_INT.py

Long-term stochastic simulation using multi-level stochastic Stuart-Landau
modeling (MSLM), enabled by DAHD analysis.

This performs the same algorithm as MSLM_FCST.py (already in the DAHD4
package) but simulates a fresh trajectory of length NE from scratch, with
rejection sampling / annealing of the nonlinear term if a realization blows
up, rather than forecasting forward from an existing time series.

Python conversion of MSLM_INT.m (Dmitri Kondrashov) by Taylor McDonald and Dmitri KOndrashov, 2026.
This translation keeps MSLM_INT.m's true constrained least-squares fit
(equality/inequality constraints on the Stuart-Landau pair coefficients),
rather than the simplified unconstrained-fit-plus-clipping approximation
used in MSLM_FCST.py -- see lsqlin_util.py.

References:
    Kondrashov D., et al. 2020 Data-adaptive harmonic analysis of oceanic
    waves and turbulent flows. Chaos, 30, doi: 10.1063/5.0012077

    Kondrashov D., et al. 2026 Accurate and robust real-time prediction of
    September Arctic sea ice. Chaos: 36, doi: 10.1063/5.0295634

    Chekroun, M. D., and D. Kondrashov, 2017: Data-adaptive harmonic spectra and multilayer Stuart-Landau models.
    Chaos, 27 (9), 093 110. doi:10.1063/1.4989400

    Kondrashov, D. et al., 2018: Multiscale Stuart-Landau emulators: Application to wind-driven ocean gyres.
    Fluids, 3 (1), 21, doi:10.3390/fluids3010021.


Original MATLAB written by Dmitri Kondrashov. Version date 7/26/26.
Please send comments and suggestions to dkondras@atmos.ucla.edu
"""

import os
import sys

import numpy as np
from scipy.linalg import lstsq, cholesky

# ── Make the DAHD4 package importable ───────────────────────────────────────
# MSLM_INT.py lives in the MSLM_L96 project folder, but reuses `center` from
# the shared DAHD4 package. Override with the DAHD4_DIR environment variable
# if that package lives somewhere else on your machine.
_DAHD4_DIR = os.environ.get("DAHD4_DIR", "/Users/taylormcdonald/DAHD4/DAHD4")
if _DAHD4_DIR not in sys.path:
    sys.path.insert(0, _DAHD4_DIR)

from center import center  # noqa: E402

from lsqlin_util import lsqlin  # noqa: E402


def MSLM_INT(NE, data, NELIN, NLEVEL, NSMT, ires, inorm, iSYM, iNL, inEQ, indrandperm):
    """
    Perform long-term simulation using multi-level stochastic Stuart-Landau
    modeling enabled by DAHD analysis.

    Args:
        NE (int): Max time steps to simulate.
        data (np.ndarray): Array of DAHCs from DAHD analysis, shape (N, nmax).
        NELIN (int): Model spec -- 1: Stuart-Landau (SL), 0: linear.
        NLEVEL (int): Number of levels.
        NSMT (int): Number of stochastic realizations.
        ires (int): Noise specification -- 0: correlated white noise,
            1: permutation of last-level regression residuals.
        inorm (int): Data normalization, 0 or 1.
        iSYM (int): Symmetry constraint for the SL model (equality).
        iNL (int): Nonlinear-coefficient-pair equality constraint for the SL model.
        inEQ (int): Nonlinear-coefficient sign inequality constraint for the SL model.
        indrandperm (np.ndarray): Random indices/noise array for the stochastic
            ensemble -- shape (nmax, NE, NT0) if ires == 0 (Gaussian draws),
            or shape (NE-1, NT0) if ires == 1 (permutation indices).

    Returns:
        tuple:
            XSM (np.ndarray): Simulated data, shape (NE, nmax, NSMT) -- or the
                raw model state XX if NSMT == 0.
            xt_res (np.ndarray): Regression residuals at the last model level.
            L (np.ndarray): Model coefficient array, shape (nmax, nmax*NLEVEL, NLEVEL).
            ENL (np.ndarray): Nonlinear (Stuart-Landau) coefficients, shape (nmax, 1).
    """
    DMD, _ = center(data)

    NLN = 1
    nmax = DMD.shape[1]
    NTT = DMD.shape[0]           # original (full) training length
    NTT_orig = NTT

    XT_RES = np.zeros((NTT_orig - 1, nmax, NLEVEL))

    # XX must hold both the training sequence (length NTT_orig) used while
    # fitting the model, and later the simulated trajectory of length NE
    # (MATLAB auto-grows the array to whichever is larger on assignment).
    NROWS = max(NTT_orig, NE)
    XX = np.zeros((NROWS, nmax, NLEVEL))

    L = np.zeros((nmax, nmax * NLEVEL, NLEVEL))
    ENL = np.zeros((nmax, 1))
    F = np.zeros((nmax, NLEVEL))
    stdr = np.zeros((nmax, NLEVEL))

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

    # ── Assemble the full companion matrix & report unstable linear modes ──
    matl = np.zeros((nmax * NLEVEL, nmax * NLEVEL))
    for nl in range(NLEVEL):
        for n in range(nmax):
            matl[n + nl * nmax, :nmax * (nl + 1)] = L[n, :nmax * (nl + 1), nl]
    for nl in range(NLEVEL - 1):
        for n in range(nmax):
            cof = stdr[n, nl] if inorm == 1 else 1.0
            matl[n + nl * nmax, n + (nl + 1) * nmax] = cof

    EL, EV = np.linalg.eig(matl)
    indel = np.where(np.real(EL) > 0)[0]
    if indel.size > 0:
        print(np.real(EL[indel]))

    xt_res = XT_RES[:NTT, :, NLEVEL - 1]
    XT_RES_sim = np.zeros((max(NTT, NE - 1, 1), XT_RES.shape[1]))

    covn = np.corrcoef(XT_RES[:NTT, :, NLEVEL - 1].T) + 1e-10 * np.eye(nmax)
    rr = cholesky(covn, lower=True)

    if NSMT == 0:
        return XX, xt_res, L, ENL

    XSM = np.zeros((NE, nmax, NSMT))
    tcount = 0
    iter_count = 0
    NSM = 0
    NSM1 = 0
    coff = 1.0

    # ── Stochastic simulation, with rejection sampling / annealing of the ──
    # ── nonlinear term if a realization blows up (NaNs).                  ──
    while NSM != NSMT:
        iter_count += 1
        tcount += 1

        for NT in range(NE - 1):
            for nl in range(NLEVEL):
                pred1 = np.hstack([np.squeeze(XX[NT, :, lvl]) for lvl in range(nl + 1)])
                tmp = L[:, :nmax * (nl + 1), nl] @ pred1

                if nl != NLEVEL - 1:
                    cof = stdr[:, nl] if inorm == 1 else np.ones(nmax)
                    nr = XX[NT, :, nl + 1] * cof
                    XX[NT + 1, :, nl] = F[:, nl] + XX[NT, :, nl] + tmp + nr
                else:
                    if ires == 0:
                        idx3 = (iter_count - 1) % indrandperm.shape[2]
                        rn = rr @ indrandperm[:nmax, NT, idx3] * stdr[:, NLEVEL - 1]
                    else:
                        idx2 = (iter_count - 1) % indrandperm.shape[1]
                        timest = int(indrandperm[NT % indrandperm.shape[0], idx2])
                        rn = XT_RES[timest % NTT, :, -1]
                    XT_RES_sim[NT, :] = rn
                    XX[NT + 1, :, nl] = F[:, nl] + XX[NT, :, nl] + tmp + rn

                if nl == 0 and NELIN == 1:
                    for n0 in range(nmax // 2):
                        indk = [2 * n0, 2 * n0 + 1]
                        for n in indk:
                            pred = XX[NT, n, 0] * (
                                XX[NT, indk[0], 0] ** 2 + XX[NT, indk[1], 0] ** 2
                            )
                            # NOTE: the original MATLAB (MSLM_INT.m) adds F(n)
                            # here a second time, in addition to the F[:, nl]
                            # term already applied above for this same step.
                            # Preserved as-is for fidelity to the source.
                            if NLN == 1:
                                XX[NT + 1, n, 0] = XX[NT + 1, n, 0] + F[n, 0] + ENL[n, 0] * pred
                            else:
                                XX[NT + 1, n, 0] = XX[NT + 1, n, 0] + F[n, 0] + coff * ENL[n, 0] * pred

        if np.sum(np.isnan(XX[:, :, 0])) == 0:
            if NLN == 0:
                NLN = 1
                coff = 1.0
            NSM1 += 1
            NSM += 1
            if inorm == 1:
                XSM[:NE, :, NSM - 1] = XX[:NE, :, 0] * stddata
            else:
                XSM[:NE, :, NSM - 1] = XX[:NE, :, 0]
        else:
            if NLN == 1:
                NLN = 0
            coff -= 0.1
            iter_count -= 1

        print(f"TOTAL SIMULATIONs {tcount} SUCCESSFULL {NSM1}")

    return XSM, xt_res, L, ENL
