import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

# Generate multivariate time series, each of them a linear combination of
# sinusoids with different period length |f0| and variance |f0Var|.
# Furthermore, red noise realized as an AR(1) process with a noise level
# |NoiseLevel| is added. The AR(1) parameters are randomly chosen in
# |ARCoeff|.

t = np.arange(1, 130).reshape(-1, 1)

f0 = 1.0 / np.array([7.5, 5.0, 2.8, 2.3])

f0Var = np.array([
    [0.4, 0.0, 0.3, 0.3],
    [0.4, 0.2, 0.4, 0.0],
    [0.3, 0.3, 0.0, 0.4],
    [0.0, 0.4, 0.4, 0.2],
    [0.2, 0.4, 0.0, 0.4],
    [0.3, 0.0, 0.4, 0.3],
])

D = f0Var.shape[0]
np.random.seed(0)
N = t.shape[0]

NoiseLevel = 0.6
ARCoeff = np.random.rand(1, D) * 0.1 + 0.55
ARVar = 1 - ARCoeff ** 2

# Combination of sinusoids
xreff = np.zeros((N, D, len(f0)))
xref  = np.zeros((N, D))
beta  = np.zeros((D, len(f0)))

for d in range(D):
    for pos in range(len(f0)):
        beta[d, pos]     = np.random.rand() * 2 * np.pi
        xreff[:, d, pos] = (
            np.sqrt(f0Var[d, pos])
            * np.sin(2 * np.pi * f0[pos] * t + beta[d, pos]).squeeze()
        )

xref  = np.squeeze(np.sum(xreff, axis=2))
xref1 = np.squeeze(xreff[:, :, 0]) / np.sqrt(0.5) * np.sqrt(1 - NoiseLevel)
xref2 = np.squeeze(xreff[:, :, 1]) / np.sqrt(0.5) * np.sqrt(1 - NoiseLevel)
xref3 = np.squeeze(xreff[:, :, 2]) / np.sqrt(0.5) * np.sqrt(1 - NoiseLevel)
xref4 = np.squeeze(xreff[:, :, 3]) / np.sqrt(0.5) * np.sqrt(1 - NoiseLevel)
xref  = xref / np.sqrt(0.5)

# AR(1) process
r = np.random.randn(N, D)
for d in range(D):
    r[:, d] = scipy.signal.lfilter(
        [np.sqrt(ARVar[0, d])], [1, -ARCoeff[0, d]], r[:, d]
    )

# Sinusoid + AR(1)
data   = np.sqrt(1 - NoiseLevel) * xref + np.sqrt(NoiseLevel) * r
noise  = np.sqrt(NoiseLevel) * r
xref   = np.sqrt(1 - NoiseLevel) * xref
signal = xref
x      = data
xmax   = 0.5
xmin   = -0.5

# Only plot when run directly, not when imported
if __name__ == '__main__':
    fig = plt.figure(1, figsize=(40 / 2.54, 40 / 2.54), dpi=100)
    plt.rcParams.update({
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.titlesize': 10,
    })

    panels = [
        (321, xref1,  '(a) Mode 1',            xmin,  xmax),
        (322, xref2,  '(b) Mode 2',             xmin,  xmax),
        (323, xref3,  '(c) Mode 3',             xmin,  xmax),
        (324, xref4,  '(d) Mode 4',             xmin,  xmax),
        (325, signal, '(e) Signal: Sum of Modes 1-4', -2, 2),
        (326, data,   '(f) Data: Signal + Noise',     -2, 2),
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