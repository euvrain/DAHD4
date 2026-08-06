"""
runMSLM_ice.py

Prediction of pan-Arctic sea ice extent using DAHD + MSLM forecasting.
Python conversion of runMSLM_ice.m by Taylor McDonald, SETI Institute, 2026.

Reference:
    Kondrashov D. et al. 2026 Accurate and robust real-time prediction of
    September Arctic sea ice. Chaos: 36, doi: 10.1063/5.0295634

Python conversion by Taylor McDonald, SETI Institute, 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import interp1d
import openpyxl

from DAHD4freq_part_weight import DAHD4freq_part_weight
from DAHM4_ex import DAHM4_ex
from MSLM_FCST import MSLM_FCST
from dahc import dahc
from hrc import hrc


# ── 1. Read Arctic Sea Ice Data ───────────────────────────────────────────────
EXCEL_FILE = 'N_Sea_Ice_Index_Regional_Daily_Data_G02135-2024June17.xlsx'
wb         = openpyxl.load_workbook(EXCEL_FILE, data_only=True)
sheet_idx  = list(range(1, 28, 2))

all_weekly = []
for si in sheet_idx:
    ws     = wb.worksheets[si]
    rows   = list(ws.iter_rows(values_only=True))
    header = rows[0]
    n_yr   = len([c for c in header[2:] if isinstance(c, (int, float))])
    num0   = []
    for row in rows[1:365]:
        num0.append([
            float(v) if isinstance(v, (int, float)) and v is not None else np.nan
            for v in row[2:2 + n_yr]
        ])
    num0   = np.array(num0, dtype=float)
    weekly = np.zeros((52, n_yr))
    for yr in range(n_yr):
        weekly[:, yr] = num0[:, yr].reshape(52, 7).mean(axis=1)
    all_weekly.append(weekly.T.reshape(-1))

max_len = max(len(a) for a in all_weekly)
extent  = np.full((max_len, 14), np.nan)
for k, a in enumerate(all_weekly):
    extent[:len(a), k] = a

# ── 2. Interpolate missing values ─────────────────────────────────────────────
data = np.copy(extent)
for kk in range(14):
    tmp  = extent[:, kk]
    indm = np.where(np.isnan(tmp))[0]
    indf = np.where(~np.isnan(tmp))[0]
    if len(indm) > 0 and len(indf) > 1:
        f = interp1d(indf, tmp[indf], bounds_error=False, fill_value='extrapolate')
        data[indm, kk] = f(indm)

nan_ind      = np.where(np.isnan(data[:, 0]))[0]
NN           = nan_ind[0] if len(nan_ind) > 0 else data.shape[0]
data         = data[:NN - 1, :]
data[-1, 13] = data[-2, 13]

# ── 3. Build regional composites ──────────────────────────────────────────────
iceCEN = data[:, 5]
iceA   = data[:, 2]  + data[:, 6]
iceB   = data[:, 4]
iceC   = data[:, 1]  + data[:, 10]
iceD   = data[:, 11] + data[:, 7]
iceE   = data[:, 0]
iceF   = data[:, 8]
iceG   = data[:, 3]  + data[:, 9] + data[:, 12] + data[:, 13]

ICET      = np.column_stack([iceA, iceB, iceC, iceD, iceE + iceF, iceG])
iceextobs = np.column_stack([iceCEN, ICET])
DD        = iceextobs.shape[1]

# ── 4. Trim to analysis period ────────────────────────────────────────────────
iceextobs = iceextobs[1404:, :]
iceext    = iceextobs.copy()
weeks     = np.arange(1, iceext.shape[0] + 1)
yearst    = 2006 + weeks / 52

# ── 5. Plot raw sea ice extent (Fig 1) ───────────────────────────────────────
n_hist  = 728 + 52 * 4
n_years = n_hist // 52
iceyear = iceextobs[:n_hist, :].reshape(n_years, 52, DD).transpose(1, 0, 2)
ice2021 = iceextobs[n_hist:, :]
tweek   = ice2021.shape[0]

titles = ['Center', 'Beaufort/Chukchi', 'Canadian Archipelago',
          'Barents/Kara', 'Laptev/East Siberian', 'Baffin/Greenland', 'Other']
ylims  = [(2, 3.5), (0, 2), (0, 1), (0, 2), (0, 2.5), (0, 2), (0, 2.5)]

wk_slice = slice(19, 40)
x_weeks  = np.arange(20, 41)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for k in range(DD):
    ax = axes.flat[k]
    ax.plot(x_weeks, iceyear[wk_slice, :, k] / 1e6, 'r', linewidth=0.5)
    ax.plot(x_weeks, iceyear[wk_slice, -2, k] / 1e6, 'b', linewidth=2)
    ax.plot(x_weeks, iceyear[wk_slice,  6, k] / 1e6, 'g', linewidth=2)
    n_cur = min(tweek - 19, len(x_weeks))
    if n_cur > 0:
        ax.plot(x_weeks[:n_cur], ice2021[19:19 + n_cur, k] / 1e6, 'k', linewidth=4)
    ax.axvline(min(tweek, 40), color='r', linestyle='--')
    for wk in [24, 28, 32, 36]:
        if tweek > wk: ax.axvline(wk, color='k', linestyle='--')
    ax.set_title(titles[k] if k < len(titles) else f'Region {k+1}')
    ax.set_xlim([20, 40]); ax.set_xlabel('week'); ax.set_ylabel('10$^6$ km$^2$')
    if k < len(ylims): ax.set_ylim(ylims[k])
    ax.tick_params(labelsize=12)

ax = axes.flat[7]
ax.plot(x_weeks, np.sum(iceyear[wk_slice, :, :], axis=2)            / 1e6, 'r', linewidth=0.5)
ax.plot(x_weeks, np.sum(iceyear[wk_slice, -2, :], axis=1)           / 1e6, 'b', linewidth=2)
ax.plot(x_weeks, np.sum(iceyear[wk_slice,  6, :], axis=1)           / 1e6, 'g', linewidth=2)
n_cur = min(tweek - 19, len(x_weeks))
if n_cur > 0:
    ax.plot(x_weeks[:n_cur], np.sum(ice2021[19:19+n_cur, :], axis=1) / 1e6, 'k', linewidth=4)
ax.set_title('pan-Arctic'); ax.set_xlim([20, 40]); ax.set_ylim([3, 13])
ax.set_xlabel('week'); ax.set_ylabel('10$^6$ km$^2$'); ax.tick_params(labelsize=12)
plt.tight_layout(); plt.show()

# ── 6. Forecast parameters ────────────────────────────────────────────────────
TARG = 767 + 4 * 52
KT   = iceext.shape[0]
LEAD = TARG - KT
KF   = KT

yearst_full = np.append(yearst,
    np.arange(1, LEAD + 1) / 52 + yearst[KF - 1])

iceext  = iceextobs[:KF, :]
icemean = np.mean(iceext, axis=0)

# ── 7. Center and remove seasonal cycle ──────────────────────────────────────
iceext_c = iceext - icemean
anom     = np.zeros_like(iceext_c)
scycle   = np.zeros((52, DD))
icycle   = 1

for j in range(DD):
    for i in range(52):
        idx           = np.arange(i, iceext_c.shape[0], 52)
        scycle[i, j]  = np.mean(iceext_c[idx, j])
        anom[idx, j]  = iceext_c[idx, j] - scycle[i, j]

X = anom

# ── 8. DAHD Analysis ──────────────────────────────────────────────────────────
W   = 39
WW  = 2 * W - 1
D   = X.shape[1]
NP  = D
NFE = W
wt  = 'bartlett'

print("Running DAHD spectral analysis...")
fE2, VP, FEP = DAHD4freq_part_weight(X, W, NFE, NP, wt)

# Fig 2 — DAHD spectrum
plt.figure()
plt.semilogy(fE2, np.abs(VP).T, 'ro', markersize=4, markerfacecolor='r')
plt.ylabel(r'$\lambda$'); plt.xlim([0, 0.5]); plt.ylim(bottom=1e6)
plt.xlabel('frequency (cycle/week)'); plt.tick_params(labelsize=14)
plt.tight_layout(); plt.show()

# ── 9. Compute DAHMs ──────────────────────────────────────────────────────────
NP = D
EP = np.zeros(((2 * W - 1) * D, 2 * NP, NFE))
print("Computing DAHMs...")
for iff in range(NFE):
    ER  = DAHM4_ex(FEP[:, :, iff], W, iff, 2 * NP)
    EP[:, :, iff] = np.reshape(ER, ((2 * W - 1) * D, 2 * NP))

# Fig 3 — DAHM modes (NF=9:16 in MATLAB 1-indexed = 8:15 in Python)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for NFFF, NF in enumerate(range(8, 16)):
    ax = axes.flat[NFFF]
    cf = ax.contourf(np.reshape(EP[:, 1, NF], (WW, D)).T, 20, cmap='jet')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'f={round(fE2[NF], 2)}')
    ax.set_yticks(range(D))
    ax.set_yticklabels(['Cen', 'BCh', 'Can', 'BaK', 'LES', 'BaB', 'Oth'][:D])
    ax.tick_params(labelsize=11)
plt.tight_layout(); plt.show()

# Fig 4 — DAHM modes (NF=1:2:15 quadrature pairs, 4x4 grid)
# Rows 0-1: first eigenvector; Rows 2-3: second eigenvector (quadrature)
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
NFFF = 0
for NF in range(0, 15, 2):
    # First eigenvector (col 0) — rows 0 and 1
    row = 0 if NF <= 6 else 1
    col = NFFF if NF <= 6 else NFFF - 4
    ax = axes[row, col]
    cf = ax.contourf(np.reshape(EP[:, 0, NF], (WW, D)).T, 20, cmap='jet')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'f={round(fE2[NF], 2)}')
    ax.set_xlabel('time (week)')
    ax.set_yticks(range(D))
    ax.set_yticklabels(['Cen', 'BCh', 'Can', 'BaK', 'LES', 'BaB', 'Oth'][:D])
    ax.tick_params(labelsize=10)
    NFFF += 1

NFFF = 0
for NF in range(0, 15, 2):
    # Second eigenvector (col 1) — rows 3 and 2 (SWAPPED to match MATLAB)
    row = 3 if NF <= 6 else 2
    col = NFFF if NF <= 6 else NFFF - 4
    ax = axes[row, col]
    cf = ax.contourf(np.reshape(EP[:, 1, NF], (WW, D)).T, 20, cmap='jet')
    plt.colorbar(cf, ax=ax)
    ax.set_title(f'f={round(fE2[NF], 2)}')
    ax.set_xlabel('time (week)')
    ax.set_yticks(range(D))
    ax.set_yticklabels(['Cen', 'BCh', 'Can', 'BaK', 'LES', 'BaB', 'Oth'][:D])
    ax.tick_params(labelsize=10)
    NFFF += 1
plt.tight_layout(); plt.show()

# ── 10. MSLM Forecast ────────────────────────────────────────────────────────
# MATLAB: NFE=20, NFS=1 (1-indexed) → Python: NFS=0, NFE_fcst=20
NFS_fcst = 0
NFE_fcst = 20
NT0      = 600   # ensemble size (MATLAB: NT0=600)
inorm    = 1
ires     = 0     # MATLAB default ires=0
iSYM     = 1
iNL      = 1
inEQ     = 1
L_lvl    = 2

NA   = X.shape[0] - EP.shape[0] // D + 1   # MATLAB: NA=size(X,1)-size(EP,1)/D+1
NE   = NA + LEAD
NXT  = NE + EP.shape[0] // D - 1

print(f"NA={NA}, NE={NE}, NXT={NXT}, LEAD={LEAD}")

RXT  = np.zeros((NXT, DD, NFE_fcst - NFS_fcst, NT0))
RRT  = np.zeros((X.shape[0], DD, NFE_fcst - NFS_fcst))
AXT  = np.zeros((NE, 2, NT0, NFE_fcst - NFS_fcst))

np.random.seed(0)   # MATLAB: rng('default')

print("Running MSLM forecast...")
for NF in range(NFS_fcst, NFE_fcst):
    print(f"  NF={NF+1}/{NFE_fcst}")

    # MATLAB: if NF==1 NM=D else NM=2*D (1-indexed)
    NM   = D if NF == 0 else 2 * D
    ncmp = NM

    ER = EP[:, :NM, NF]
    A  = dahc(X, ER)

    # HRC reconstruction for variance check
    indm = np.arange(ncmp)
    RR_f = hrc(A, ER, DD, indm)
    n    = min(RR_f.shape[0], X.shape[0])
    RRT[:n, :, NF - NFS_fcst] = RR_f[:n, :]

    DMD = A[:, indm]

    # Build random permutation array
    np.random.seed(0)   # rng('default') inside loop in MATLAB
    if ires == 0:
        indrandperm = np.random.randn(ncmp, NE, NT0)
    else:
        indrandperm = np.zeros((NA - 1, NT0), dtype=int)
        for nt in range(NT0):
            indrandperm[:, nt] = np.random.permutation(NA - 1)

    # Call MSLM_FCST with correct parameters
    if NF == 0:
        xx, xt_res, LL, _ = MSLM_FCST(
            LEAD, DMD, 0, L_lvl, NT0, ires, inorm,
            0, 0, 0, indrandperm
        )
    else:
        xx, xt_res, LL, ENL = MSLM_FCST(
            LEAD, DMD, 1, L_lvl, NT0, ires, inorm,
            iSYM, iNL, inEQ, indrandperm
        )

    AXT[:, :2, :, NF - NFS_fcst] = xx[:NE, :2, :]

    # HRC reconstruction for each ensemble member
    RXZ = np.zeros((NXT, DD, NT0))
    for KK in range(NT0):
        RXZ[:min(xx.shape[0], NXT), :, KK] = hrc(
            xx[:, indm, KK], ER[:, indm], DD,
            np.arange(A[:, indm].shape[1])
        )[:min(xx.shape[0], NXT), :]

    RXT[:, :, NF - NFS_fcst, :] = RXZ

# ── 11. Post-processing ───────────────────────────────────────────────────────
NET  = X.shape[0]   # forecast start index
indf = np.arange(NFE_fcst - NFS_fcst)

# Sum RRT across frequencies and add seasonal cycle + mean
RR = np.sum(RRT[:, :, indf], axis=2)   # (NET, DD)

# Sum RXT across frequencies: (NXT, DD, NT0)
RX = np.sum(RXT[:, :, indf, :], axis=2)   # (NXT, DD, NT0)

# Add seasonal cycle back
if icycle == 1:
    for j in range(DD):
        for i in range(52):
            idx = np.arange(i, RR.shape[0], 52)
            idx = idx[idx < RR.shape[0]]
            RR[idx, j] += scycle[i, j]
            for k in range(NT0):
                idx2 = np.arange(i, RX.shape[0], 52)
                idx2 = idx2[idx2 < RX.shape[0]]
                RX[idx2, j, k] += scycle[i, j]

RX0 = RX.copy()

# Add icemean
tmp = icemean   # (DD,)
for k in range(NT0):
    RX0[:, :, k] += tmp

# Zero out negative SIE
RX0[RX0 < 0] = 0

# Subtract icemean to get anomalies back
RX00 = RX0.copy()
for k in range(NT0):
    RX00[:, :, k] -= tmp

# Regional predictions
RXR   = RX.copy()
RXRM  = np.mean(RXR, axis=2)   # (NXT, DD)

# Pan-Arctic sums
RR_pan  = np.sum(RR, axis=1)         # (NET,)
RXX     = np.squeeze(np.sum(RX, axis=1))  # (NXT, NT0)
RX_mean = np.sum(np.mean(RX, axis=2), axis=1)  # (NXT,)

# RXRC: sum over freq of RXT → (NXT, DD, NT0) already = RX
RXRC  = np.sum(RXT[:, :, indf, :], axis=2)   # (NXT, DD, NT0)
RXC   = np.squeeze(np.sum(RXRC, axis=1))      # (NXT, NT0)
RXRMC = np.squeeze(np.sum(np.mean(RXT[:, :, indf, :], axis=3), axis=2))  # (NXT, DD)
RXMC  = np.sum(RXRMC, axis=1)                 # (NXT,)

# ── 12. Load observations and plot forecast (Fig 5) ──────────────────────────
mat    = scipy.io.loadmat('ROBSF.mat')
ROBSF  = mat['ROBSF'].ravel()
ROBSRF = mat['ROBSRF']

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
ylims2 = [(-0.15, 0.15), (-0.3, 0.3), (-0.2, 0.2),
          (-0.3, 0.3),   (-0.3, 0.3), (-0.3, 0.3), (-0.4, 0.4)]

lmax = RXRC.shape[0]
t_full = yearst_full[:lmax]

for kk in range(DD):
    ax = axes.flat[kk]
    # Cyan: ensemble members
    ax.plot(t_full[NET:NET + LEAD + 1],
            np.squeeze(RXRC[NET:NET + LEAD + 1, kk, :]) / 1e6,
            'c-', linewidth=1)
    # Red: ensemble mean
    if NET != NET + LEAD:
        ax.plot(t_full[NET:NET + LEAD + 1],
                RXRMC[NET:NET + LEAD + 1, kk] / 1e6,
                'r.-', markersize=8, linewidth=2)
    # Black: observations
    ax.plot(t_full[:len(ROBSRF)], ROBSRF[:, kk] / 1e6,
            'k.-', markersize=8, linewidth=2)
    ax.set_title(titles[kk] if kk < len(titles) else f'Region {kk+1}')
    if kk < len(ylims2): ax.set_ylim(ylims2[kk])
    tick_idx = [18 * 52 + 20, 18 * 52 + 30, 18 * 52 + 39]
    valid    = [i for i in tick_idx if i < len(t_full)]
    ax.set_xticks([t_full[i] for i in valid])
    ax.set_xticklabels(['20', '30', '40'][:len(valid)])
    ax.set_xlabel('weeks'); ax.set_ylabel('10$^6$ km$^2$')
    ax.tick_params(labelsize=12)

ax = axes.flat[7]
ax.plot(t_full[NET:NET + LEAD + 1],
        RXC[NET:NET + LEAD + 1, :] / 1e6, 'c-', linewidth=1)
ax.plot(t_full[NET:NET + LEAD + 1],
        RXMC[NET:NET + LEAD + 1]   / 1e6, 'r.-', markersize=8, linewidth=2)
ax.plot(t_full[:len(ROBSF)], ROBSF / 1e6,
        'k.-', markersize=8, linewidth=2)
ax.set_title('pan-Arctic'); ax.set_ylim([-1, 1])
ax.set_xlabel('weeks'); ax.set_ylabel('10$^6$ km$^2$')
ax.tick_params(labelsize=12)
plt.tight_layout(); plt.show()

# ── 13. September prediction summary ─────────────────────────────────────────
sept   = np.arange(712 + 5 * 52 - 1, 715 + 5 * 52)
sept   = sept[sept < RXRC.shape[0]]

tmppp  = np.squeeze(np.sum(RX0, axis=1))       # (NXT, NT0) — sum regions
tmppp1 = tmppp + np.sum(icemean)                # add mean
tmppp2 = np.mean(tmppp1[sept, :], axis=0) / 1e6
sice   = np.mean(tmppp2)
uns    = np.std(RXX, axis=1, ddof=1)

print(f'SeptSIE (Mkm2): {sice:.4f}')
print(f'Ensemble spread (Mkm2): {np.mean(uns[sept]) / 1e6:.4f}')

# ── 14. Histogram ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(tmppp2, bins=40, edgecolor='black', alpha=0.7)
ax.axvline(sice, color='r', linestyle='--', linewidth=2,
           label=f'Prediction: {sice:.2f}')
ax.axvline(4.40, color='k', linestyle='-', linewidth=2,
           label='Reference: 4.40')
ax.set_xlim([4.2, 5.0]); ax.set_ylim([0, 45])
ax.set_xlabel('10$^6$ km$^2$'); ax.set_ylabel('Count')
ax.set_title('June'); ax.legend(); ax.tick_params(labelsize=14)
plt.tight_layout(); plt.show()