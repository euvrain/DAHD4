import numpy as np
import matplotlib.pyplot as plt
from time import time
from center import center
from dahc import dahc
from hrc import hrc
from DAHD4freq_part_weight import DAHD4freq_part_weight
from DAHM4_ex import DAHM4_ex
from generate_data import x, xref1, xref2, xref3, xref4, N, signal, data, xmin, xmax

# ── 1. Plot input data (Fig 1) ────────────────────────────────────────────────
fig1 = plt.figure(1, figsize=(40 / 2.54, 40 / 2.54), dpi=100)
plt.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
})
panels = [
    (321, xref1,  '(a) Mode 1',                   xmin, xmax),
    (322, xref2,  '(b) Mode 2',                   xmin, xmax),
    (323, xref3,  '(c) Mode 3',                   xmin, xmax),
    (324, xref4,  '(d) Mode 4',                   xmin, xmax),
    (325, signal, '(e) Signal: Sum of Modes 1-4', -2,   2),
    (326, data,   '(f) Data: Signal + Noise',     -2,   2),
]
for sp, arr, title, vmin, vmax in panels:
    plt.subplot(sp)
    cf = plt.contourf(arr.T, 20, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(cf)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    plt.gca().tick_params(labelsize=10)
plt.tight_layout()
plt.show()

# ── 2. Center data ────────────────────────────────────────────────────────────
fpcs, _ = center(x)
W   = 65
WW  = 2 * W - 1
D   = fpcs.shape[1]
NFE = W

# ── 3. DAHD spectrum NO weighting (Fig 2) ────────────────────────────────────
start = time()
fE2_nw, VP_nw, _ = DAHD4freq_part_weight(fpcs, W, NFE, D, 'none')
print(f"Spectrum (no weight) elapsed: {time() - start:.4f} seconds")

NP = 1
fig2 = plt.figure(2)
plt.rcParams.update({'font.size': 16})
plt.semilogy(fE2_nw[1:], np.abs(VP_nw[:, 1:]).T, 'ro', markersize=4, markerfacecolor='r')
plt.semilogy(fE2_nw[17], np.abs(VP_nw[:NP, 17]), 'bo', markersize=8, markerfacecolor='b')
plt.semilogy(fE2_nw[26], np.abs(VP_nw[:NP, 26]), 'co', markersize=8, markerfacecolor='c')
plt.semilogy(fE2_nw[46], np.abs(VP_nw[:NP, 46]), 'go', markersize=8, markerfacecolor='g')
plt.semilogy(fE2_nw[56], np.abs(VP_nw[:NP, 56]), 'ko', markersize=8, markerfacecolor='k')
plt.ylabel(r'$\lambda$')
plt.xlim([0, 0.5])
plt.ylim(bottom=1e-2)
plt.xlabel('Freq')
plt.tight_layout()
plt.show()

# ── 4. DAHD spectrum WITH Hamming weighting (Fig 3) ──────────────────────────
start = time()
fE2, VP, FEP = DAHD4freq_part_weight(fpcs, W, NFE, D, 'hamming')
print(f"Spectrum (hamming) elapsed: {time() - start:.4f} seconds")

fig3 = plt.figure(3)
plt.semilogy(fE2[1:], np.abs(VP[:, 1:]).T, 'ro', markersize=4, markerfacecolor='r')
plt.semilogy(fE2[17], np.abs(VP[:NP, 17]), 'bo', markersize=8, markerfacecolor='b')
plt.semilogy(fE2[26], np.abs(VP[:NP, 26]), 'co', markersize=8, markerfacecolor='c')
plt.semilogy(fE2[46], np.abs(VP[:NP, 46]), 'go', markersize=8, markerfacecolor='g')
plt.semilogy(fE2[56], np.abs(VP[:NP, 56]), 'ko', markersize=8, markerfacecolor='k')
plt.ylabel(r'$\lambda$')
plt.xlim([0, 0.5])
plt.ylim(bottom=1e-2)
plt.xlabel('Freq')
plt.tight_layout()
plt.show()

# ── 5. Compute DAHMs (using Hamming-weighted FEP) ────────────────────────────
EP = np.zeros(((2 * W - 1) * D, 2 * NP, NFE))
start = time()
for iff in range(NFE):
    ER  = DAHM4_ex(FEP[:, :, iff], W, iff, 2 * NP)
    ERR = np.reshape(ER, ((2 * W - 1) * D, 2 * NP))
    EP[:, :, iff] = ERR
print(f"DAHMs elapsed: {time() - start:.4f} seconds")

# ── 6. Plot DAHMs (Fig 4) ─────────────────────────────────────────────────────
tt = ['(a) f_1', '(b) f_2', '(c) f_3', '(d) f_4']
ff = [17, 26, 46, 56]

fig4 = plt.figure(4, figsize=(20 / 2.54, 40 / 2.54), dpi=100)
plt.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})
kk = 0
for pos in range(4):
    for K in range(2):
        kk += 1
        plt.subplot(4, 2, kk)
        cf = plt.contourf(
            np.reshape(EP[:, K, ff[pos]], (WW, 6)).T,
            20, cmap='jet'
        )
        cbar = plt.colorbar(cf)
        cbar.locator = plt.MaxNLocator(nbins=5)
        cbar.update_ticks()
        if K == 0:
            plt.title(tt[pos])
        plt.xlabel('Time')
        plt.ylabel('Space')
        plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        plt.xticks([0, 20, 40, 60, 80, 100, 120],
                   [1, 20, 40, 60, 80, 100, 120])
plt.subplots_adjust(hspace=1.0, wspace=0.5)
plt.show()

# ── 7. Reconstruction ─────────────────────────────────────────────────────────
ifff = [17, 26, 46, 56]
NE   = fpcs.shape[0]
NPP  = 1
PCF  = np.zeros((NE, D, len(ifff)))
AF   = np.zeros((N - (2 * W - 1) + 1, 2, len(ifff)))

for i in range(len(ifff)):
    iff          = ifff[i]
    tmp          = np.squeeze(EP[:, :2 * NPP, iff])
    A            = dahc(fpcs, tmp)
    AF[:, :, i]  = A
    PCF[:, :, i] = hrc(A, tmp, fpcs.shape[1], np.arange(2 * NPP))

R1 = np.squeeze(PCF[:, :, 0])
R2 = np.squeeze(PCF[:, :, 1])
R3 = np.squeeze(PCF[:, :, 2])
R4 = np.squeeze(PCF[:, :, 3])

# ── 8. Normalized RMSE ───────────────────────────────────────────────────────
print('Normalized RMSE of reconstruction:')
for label, ref, rec in zip(['f_1', 'f_2', 'f_3', 'f_4'],
                            [xref1, xref2, xref3, xref4],
                            [R1, R2, R3, R4]):
    rmse = np.sum((ref - rec) ** 2) / np.sum(ref ** 2)
    print(f"  {label}: {rmse:.4f}")

# ── 9. Plot HRC reconstructions (Fig 5) ──────────────────────────────────────
fig5 = plt.figure(5, figsize=(20 / 2.54, 40 / 2.54), dpi=100)
plt.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 8,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})
pairs = [
    (xref1, R1, 'f_1'),
    (xref2, R2, 'f_2'),
    (xref3, R3, 'f_3'),
    (xref4, R4, 'f_4'),
]
kk = 0
for ref, rec, label in pairs:
    for arr, title in [(ref, f'Mode at {label}'), (rec, f'Recons at {label}')]:
        kk += 1
        plt.subplot(4, 2, kk)
        cf = plt.contourf(arr.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
        cbar = plt.colorbar(cf)
        cbar.locator = plt.MaxNLocator(nbins=5)
        cbar.update_ticks()
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Space')
        plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
        plt.xticks([0, 20, 40, 60, 80, 100],
                   [1, 20, 40, 60, 80, 100])
plt.subplots_adjust(hspace=1.0, wspace=0.5)
plt.show()