"""
runmsmDAHD4_L96.py

Top-level driver: DAHD spectral analysis + multi-level stochastic
Stuart-Landau (MSLM) simulation of the Lorenz-96 model (L96_F6.mat).

Python conversion of runmsmDAHD4_L96.m (Dmitri Kondrashov) by
Taylor McDonald and Dmitri Kondrashov 2026,  reusing the shared DAHD4 package
(center, dahc, hrc, DAHD4freq_part_weight, DAHM4_ex) plus the new
MSLM_INT.py / statglob.py translated alongside this script.

The original MATLAB used `parfor` for the per-frequency MSLM loop; this
translation runs it serially (the iterations are independent, so it can be
parallelized with e.g. `joblib.Parallel` if desired -- not done here to keep
the translation straightforward).
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# ── Make the DAHD4 package importable ───────────────────────────────────────
_DAHD4_DIR = os.environ.get("DAHD4_DIR", "/Users/dmitri/PycharmProjects/DAHD4")
if _DAHD4_DIR not in sys.path:
    sys.path.insert(0, _DAHD4_DIR)

from center import center  # noqa: E402
from dahc import dahc  # noqa: E402
from hrc import hrc  # noqa: E402
from DAHD4freq_part_weight import DAHD4freq_part_weight  # noqa: E402
from DAHM4_ex import DAHM4_ex  # noqa: E402

from MSLM_INT import MSLM_INT
from statglob import statglob


# ── 1. Load & prepare L96 data ──────────────────────────────────────────────
mat = scipy.io.loadmat('L96_F6.mat')
data_raw = mat['data']

K = 10
data_ds = data_raw[::K, :]
data_c, _ = center(data_ds)
X = data_c

W = 160
D = X.shape[1]
NE = X.shape[0]
NFE = W
NP = D
wt = 'bartlett'

print("Running DAHD spectral analysis...")
fE2, VP, FEP = DAHD4freq_part_weight(X, W, NFE, NP, wt)

# ── 2. Spectrum plot ─────────────────────────────────────────────────────────
# NCMP =  int(round(0.5 * D))
NCMP = 20
NFEE = 30  # number of frequencies to emulate
ff = [1, 9, 19, 29]  # (MATLAB ff = [2,10,20,30], 1-indexed)

plt.figure()
plt.semilogy(fE2, np.abs(VP).T, 'ro', markersize=4, markerfacecolor='r')
plt.semilogy(fE2[1:NFEE], np.abs(VP[:int(NCMP), 1:NFEE]).T, 'bo',
             markersize=4, markerfacecolor='b')
for pos in ff:
    plt.semilogy(fE2[pos], np.abs(VP[0, pos]), 'ko', markersize=8, markerfacecolor='k')
plt.ylabel(r'$\lambda$')
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.show()

# ── 3. Compute DAHMs (space-time modes) ─────────────────────────────────────
NP = D
EP = np.zeros(((2 * W - 1) * D, 2 * NP, NFE))
print("Computing DAHMs...")
for iff in range(NFE):
    ER = DAHM4_ex(FEP[:, :, iff], W, iff, 2 * NP)
    EP[:, :, iff] = np.reshape(ER, ((2 * W - 1) * D, 2 * NP))

# ── 4. Mode contour plots + DAHC coefficient time series (4 selected freqs) ─
fig, axes = plt.subplots(4, 3, figsize=(15, 18))
for pos in range(4):
    for Kc in range(2):
        ax = axes[pos, Kc]
        cf = ax.contourf(EP[:, Kc, ff[pos]].reshape(2 * W - 1, D).T, 20, cmap='jet')
        plt.colorbar(cf, ax=ax)
        if Kc == 0:
            ax.set_title(f'f={fE2[ff[pos]]:.4g}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Space')
        ax.tick_params(labelsize=16)

for pos in range(4):
    ER4 = EP[:, 0:2, ff[pos]]
    A = dahc(X, ER4)
    ax = axes[pos, 2]
    ax.plot(A, linewidth=2)
    ax.set_xlim([0, 300])
    ax.set_xlabel('Time')
    ax.set_title(f'f={fE2[ff[pos]]:.4g}')
    ax.tick_params(labelsize=16)
plt.tight_layout()
plt.show()

# ── 5. MSLM simulation, per frequency band ──────────────────────────────────
NFS = 0
NFE = NFEE
NT0 = 2

NEX = X.shape[0]
RXT = np.zeros((NEX, D, NFE, NT0))
RRT = np.zeros((NEX, D, NFE))

WW = 2 * W - 1
NA = X.shape[0] - WW + 1
NE = NEX - WW + 1

print("Running MSLM simulation per frequency band...")
for NF in range(NFS, NFE):
    print(f'FREQ={NF + 1}')

    if NF == 0:
        NM = D
        ncmp = D // 2
    else:
        NM = D * 2
        # (original MATLAB reassigns ncmp several times; only the final
        # value -- ncmp = 2*NCMP -- is actually used)
        ncmp = int(round(2 * NCMP))

    ER = EP[:, :NM, NF]
    A = dahc(X, ER)

    indm = np.arange(ncmp)
    RR = hrc(A, ER, D, indm)
    RRT[:, :, NF] = RR

    DMD = A[:, indm]
    inorm = 1
    # (original MATLAB sets ires=0 then immediately overwrites it with
    # ires=1; only the final value takes effect)
    ires = 0
    L_lvl = 5

    np.random.seed(0)  # rng('default')
    if ires == 0:
        indrandperm = np.random.randn(D * 2, NE, NT0)
    else:
        indrandperm = np.zeros((NE - 1, NT0), dtype=int)
        for NT in range(NT0):
            indrandperm[:, NT] = np.random.permutation(NE - 1)

    if NF == 0:
        xx, xt_res, LL, ENL = MSLM_INT(NE, DMD, 0, L_lvl, NT0, ires, inorm,
                                        0, 0, 0, indrandperm)
    else:
        iNL, iSYM, inEQ = 1, 1, 1
        xx, xt_res, LL, ENL = MSLM_INT(NE, DMD, 1, L_lvl, NT0, ires, inorm,
                                        iSYM, iNL, inEQ, indrandperm)
        #print(ENL.T)
    RXZ = np.zeros((NEX, D, NT0))
    for KK in range(NT0):
        indf = indm
        RXZ[:, :, KK] = hrc(xx[:, indf, KK], ER[:, indf], D, np.arange(len(indf)))

    RXT[:, :, NF, :] = RXZ

# ── 6. Reconstruction vs. MSLM comparison plots ─────────────────────────────
NET = 2000
RR = np.sum(RRT, axis=2)

#fig, (ax_top, ax_bot) = plt.subplots(3, 1, figsize=(10, 12))
fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # Adjust figsize as needed
for KK in range(RXT.shape[3]):
    RX = np.sum(RXT[:, :, :, KK], axis=2)
    print(f"Simulation #{KK + 1} "
          f"Var Recons: {np.sum(np.var(RR, axis=0, ddof=1)):.6g} "
          f"Var MSLM: {np.sum(np.var(RX, axis=0, ddof=1)):.6g}")

    indt = np.arange(max(NEX - NET - 1, 0), NEX)
    cmin, cmax = -8, 8
    levels = np.linspace(cmin, cmax, 21)

    # cf1 = ax_bot.contourf(RX[indt, :].T, levels=levels, cmap='jet', extend='both')
    # plt.colorbar(cf1, ax=ax_bot)
    # ax_bot.set_title('MSLM'); ax_bot.set_xlabel('Time'); ax_bot.set_ylabel('Space')
    # ax_bot.tick_params(labelsize=16)
    #
    # cf2 = ax_top.contourf(RR[indt, :].T, levels=levels, cmap='jet', extend='both')
    # plt.colorbar(cf2, ax=ax_top)
    # ax_top.set_title('Reconstruction'); ax_top.set_xlabel('Time'); ax_top.set_ylabel('Space')
    # ax_top.tick_params(labelsize=16)
    # Subplot 311 - Full Data
    cf1 = axs[0].contourf(X[indt, :].T, levels=20, cmap='jet')
    axs[0].set_title('Full Data', fontsize=16)
    axs[0].set_xlabel('Time', fontsize=16)
    axs[0].set_ylabel('Space', fontsize=16)
    axs[0].tick_params(labelsize=16)
    fig.colorbar(cf1, ax=axs[0])
    cf1.set_clim(cmin, cmax)

    # Subplot 312 - DAHD Reconstruction
    cf2 = axs[1].contourf(RR[indt, :].T, levels=20, cmap='jet')
    axs[1].set_title('DAHD Reconstruction', fontsize=16)
    axs[1].set_xlabel('Time', fontsize=16)
    axs[1].set_ylabel('Space', fontsize=16)
    axs[1].tick_params(labelsize=16)
    fig.colorbar(cf2, ax=axs[1])
    cf2.set_clim(cmin, cmax)

    # Subplot 313 - MSLM
    cf3 = axs[2].contourf(RX[indt, :].T, levels=20, cmap='jet')
    axs[2].set_title('MSLM', fontsize=16)
    axs[2].set_xlabel('Time', fontsize=16)
    axs[2].set_ylabel('Space', fontsize=16)
    axs[2].tick_params(labelsize=16)
    fig.colorbar(cf3, ax=axs[2])
    cf3.set_clim(cmin, cmax)

plt.tight_layout()
plt.show()

if NFE - NFS > 1:
    var_RRT = np.sum(np.var(RRT, axis=0, ddof=1), axis=0)          # (NFE,)
    var_RXT = np.sum(np.var(RXT, axis=0, ddof=1), axis=0)          # (NFE, NT0)

    plt.figure()
    plt.plot(var_RRT, 'r', linewidth=2, label='Recons')
    for k in range(var_RXT.shape[1]):
        plt.plot(var_RXT[:, k], 'b', linewidth=2, label='MSLM' if k == 0 else None)
    plt.legend(loc='upper left')
    plt.xlabel('Freq #')
    plt.ylabel('variance')
    plt.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()

# ── 7. Autocorrelation / PDF diagnostics across the ensemble ────────────────
if RXT.shape[3] > 1:
    RX_all = np.sum(RXT, axis=2)                       # (NEX, D, NT0)
    datas_for_stat = RX_all.reshape(NEX, D * NT0)       # collapse to (N, K), matching
                                                         # MATLAB's implicit size() flattening
    statglob(RR, datas_for_stat, 300)
