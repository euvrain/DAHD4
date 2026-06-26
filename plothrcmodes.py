import matplotlib.pyplot as plt
from generate_data import xref1, xref2, xref3, xref4
from dahd4example import R1, R2, R3, R4
# To plot data as an image, replace:
#   contourf -> imshow

xmax = 0.5
xmin = -0.5

fig = plt.figure(figsize=(20 / 2.54, 40 / 2.54), dpi=100)
plt.rcParams.update({
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

# -- Reconstructed components --

plt.subplot(422)
cf = plt.contourf(R1.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Recons at f_1')
plt.xlim([1, 129])

plt.subplot(424)
cf = plt.contourf(R2.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Recons at f_2')
plt.xlim([1, 129])

plt.subplot(426)
cf = plt.contourf(R3.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Recons at f_3')
plt.xlim([1, 129])

plt.subplot(428)
cf = plt.contourf(R4.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Recons at f_4')
plt.xlabel('Time')
plt.xlim([1, 129])

# -- Reference modes --

plt.subplot(421)
cf = plt.contourf(xref1.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Mode at f_1')
plt.xlim([1, 129])

plt.subplot(423)
cf = plt.contourf(xref2.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Mode at f_2')
plt.xlim([1, 129])

plt.subplot(425)
cf = plt.contourf(xref3.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Mode at f_3')
plt.xlim([1, 129])

plt.subplot(427)
cf = plt.contourf(xref4.T, 20, cmap='jet', vmin=xmin, vmax=xmax)
plt.colorbar(cf)
plt.title('Mode at f_4')
plt.xlabel('Time')
plt.xlim([1, 129])

plt.tight_layout()
plt.show()