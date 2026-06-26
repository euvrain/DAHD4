import numpy as np
import matplotlib.pyplot as plt
from time import time
from center import center
from dahc import dahc
from hrc import hrc
from DAHD4freq_part_weight import DAHD4freq_part_weight
from DAHM4_ex import DAHM4_ex
from generate_data import x, xref1, xref2, xref3, xref4, N

# Center data
fpcs, _ = center(x)

W = int((fpcs.shape[0] + 1) / 2)   # maximum window length
W = 65
WW = 2 * W - 1
D = fpcs.shape[1]
NP = D      # maximum is D
NFE = W     # maximum is W

# wt = 'hann'
# wt = 'blackman'
# wt = 'hanning'
# wt = 'hamming'
# wt = 'bartlett'
wt = 'hamming'
# wt = 'none'

start = time()
fE2, VP, FEP = DAHD4freq_part_weight(fpcs, W, NFE, NP, wt)
print(f"Elapsed time: {time() - start:.4f} seconds")

# DAHD spectrum
NP = 1
plt.figure()
plt.rcParams.update({'font.size': 16})
plt.semilogy(fE2[1:], np.abs(VP[:, 1:]), 'ro', markersize=4, markerfacecolor='r')
plt.semilogy(fE2[17], np.abs(VP[:NP, 17]), 'bo', markersize=8, markerfacecolor='b')
plt.semilogy(fE2[26], np.abs(VP[:NP, 26]), 'co', markersize=8, markerfacecolor='c')
plt.semilogy(fE2[46], np.abs(VP[:NP, 46]), 'go', markersize=8, markerfacecolor='g')
plt.semilogy(fE2[56], np.abs(VP[:NP, 56]), 'ko', markersize=8, markerfacecolor='k')
plt.ylabel(r'$\lambda$')
plt.xlim([0, 0.5])
plt.ylim([1e-2, np.inf])
plt.xlabel('Freq')
plt.show()

# Compute DAHMs
EP = np.zeros(((2 * W - 1) * D, 2 * NP, NFE))
start = time()
for iff in range(NFE):
    ER = DAHM4_ex(FEP[:, :, iff], W, iff, 2 * NP)
    ERR = np.reshape(ER, ((2 * W - 1) * D, 2 * NP))
    EP[:, :, iff] = ERR
print(f"Elapsed time: {time() - start:.4f} seconds")

tt = ['(a) f_1', '(b) f_2', '(c) f_3', '(d) f_4']
ff = [17, 26, 46, 56]   # 0-indexed (MATLAB: 18, 27, 47, 57)

fig = plt.figure(figsize=(20 / 2.54, 40 / 2.54), dpi=100)
plt.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
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
        plt.colorbar(cf)
        if K == 0:
            plt.title(tt[pos])
        plt.xlabel('Time')
        plt.ylabel('Space')

plt.tight_layout()
plt.show()

# Reconstruction
# Indices of frequencies of the harmonic modes for maximum window length
ifff = [17, 26, 46, 56]   # 0-indexed (MATLAB: 18, 27, 47, 57)
NE = fpcs.shape[0]
PCF = np.zeros((NE, D, len(ifff)))
AF = np.zeros((N - (2 * W - 1) + 1, 2, len(ifff)))
NPP = 1     # reconstruct only with modes at spectral peak

for i in range(len(ifff)):
    iff = ifff[i]
    tmp = np.squeeze(EP[:, :2 * NPP, iff])
    A = dahc(fpcs, tmp)
    AF[:, :, i] = A
    PCF[:, :, i] = hrc(A, tmp, fpcs.shape[1], np.arange(2 * NPP))

R1 = np.squeeze(PCF[:, :, 0])
R2 = np.squeeze(PCF[:, :, 1])
R3 = np.squeeze(PCF[:, :, 2])
R4 = np.squeeze(PCF[:, :, 3])

# Normalized RMSE of reconstruction
print('Normalized RMSE of reconstruction:')
print(np.sum(np.sum((xref1 - R1) ** 2, axis=1)) / np.sum(np.sum(xref1 ** 2, axis=1)))
print(np.sum(np.sum((xref2 - R2) ** 2, axis=1)) / np.sum(np.sum(xref2 ** 2, axis=1)))
print(np.sum(np.sum((xref3 - R3) ** 2, axis=1)) / np.sum(np.sum(xref3 ** 2, axis=1)))
print(np.sum(np.sum((xref4 - R4) ** 2, axis=1)) / np.sum(np.sum(xref4 ** 2, axis=1)))

# Plot reconstructed harmonic components (see plothrcmodes.py)
exec(open('plothrcmodes.py').read())