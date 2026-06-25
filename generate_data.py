import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt
# %Generate time series
# Generate multivariate time series, each of them a linear combination of
# sinusoids with different period length |f0| and variance |f0Var|.
# Furthermore, red noise realized as an AR(1) process with a noise level
# |NoiseLevel| is added. The AR(1) parametes are randomly chosen in
# |ARCoeff|.

# %clear all;

t=np.arange[1:10029].reshape(-1, 1)
t=np.arange[1:1029].reshape(-1, 1)
t=np.arange[1:129].reshape(-1, 1)
f0 = 1.0/np.array([7.5, 5.0, 2.8,  2.3])




f0Var = np.array([
         [0.4,  0.0, 0.3,  0.3],         
         [0.4,  0.2, 0.4,  0.0],
         [0.3,  0.3, 0.0,  0.4],
         [0.0,  0.4, 0.4,  0.2],
         [0.2,  0.4, 0.0,  0.4],
         [0.3,  0.0, 0.4,  0.3]])

D = f0Var.shape[0]
np.random.seed(0)
N=t.length

NoiseLevel=0.6
#NoiseLevel=0.0

ARCoeff=np.random.rand(1,D)*0.1+0.55
ARVar=1-ARCoeff**2

#combination of sinusoids
xreff=np.zeros((N,D,len(f0)))
xref=np.zeros((N,D))
beta=np.zeros((D,len(f0)))
for d in range(D):
    for pos in range(len(f0)):
        beta[d,pos]=np.random.rand(1)*2*np.pi
        xreff[:,d,pos]=np.sqrt(f0Var[d,pos])*np.sin(2*np.pi*f0[pos]*t+beta[d,pos])
xref=np.squeeze(np.sum(xreff,axis=2))

#########################
xref1=np.squeeze(xreff[:,:,0])/np.sqrt(0.5)*np.sqrt(1-NoiseLevel)
xref2=np.squeeze(xreff[:,:,1])/np.sqrt(0.5)*np.sqrt(1-NoiseLevel)  
xref3=np.squeeze(xreff[:,:,2])/np.sqrt(0.5)*np.sqrt(1-NoiseLevel)  
xref4=np.squeeze(xreff[:,:,3])/np.sqrt(0.5)*np.sqrt(1-NoiseLevel)     
#########################
xref=xref/np.sqrt(0.5)
#########################

#AR(1) process
r=np.random.randn(N,D)
for d in range(D):
    r[:,d]=scipy.signal.lfilter([np.sqrt(ARVar[d])],[1, -ARCoeff[d]],r[:,d])

#sinusoid + AR(1)
data=np.sqrt(1-NoiseLevel)*xref + np.sqrt(NoiseLevel)*r
noise=np.sqrt(NoiseLevel)*r
xref=np.sqrt(1-NoiseLevel)*xref

sinal=xref
x=data

xmax=0.5
xmin=-.05
iplot=1

if iplot == 1:
    fig = plt.figure(figsize=(40/2.54, 40/2.54),  # centimeters to inches
                     dpi=100)
    plt.rcParams.update({
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.labelsize': 10,
        'axes.titlesize': 10
    })

    plt.subplot(321)
    # plt.gca().tick_params(labelsize=16)
    cf = plt.contourf(xref1.T, 20, cmap='jet')
    plt.colorbar(cf)
    cf.set_clim(xmin, xmax)
    plt.title('(a) Mode 1')
    # plt.title('(a)$x_i^1$', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.gca().tick_params(labelsize=16)

    plt.subplot(322)
    plt.gca().tick_params(labelsize=16)
    cf = plt.contourf(xref2.T, 20, cmap='jet')
    plt.colorbar(cf)
    cf.set_clim(xmin, xmax)
    plt.title('(b) Mode 2')
    # plt.title('(b)$x_i^2$', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.gca().tick_params(labelsize=16)

    plt.subplot(323)
    plt.gca().tick_params(labelsize=16)
    cf = plt.contourf(xref3.T, 20, cmap='jet')
    plt.colorbar(cf)
    cf.set_clim(xmin, xmax)
    plt.title('(c) Mode 3')
    # plt.title('(c)$x_i^3$', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.gca().tick_params(labelsize=16)

    plt.subplot(324)
    plt.gca().tick_params(labelsize=16)
    cf = plt.contourf(xref4.T, 20, cmap='jet')
    plt.colorbar(cf)
    cf.set_clim(xmin, xmax)
    plt.title('(d) Mode 4')
    # plt.title('(d)$x_i^4$', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.gca().tick_params(labelsize=16)

    plt.subplot(325)
    plt.gca().tick_params(labelsize=16)
    cf = plt.contourf(signal.T, 20, cmap='jet')
    plt.colorbar(cf)
    cf.set_clim(-2, 2)
    plt.title('(e) Signal: Sum of Modes 1-4')
    # plt.title('(e) Signal: $\\sum_{j=1}^{j=4}x_i^j$', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Space')
    plt.gca().tick_params(labelsize=16)

    plt.subplot(326)
    plt.gca().tick_params(labelsize=16)
    cf = plt.contourf(data.T, 20, cmap='jet')
    plt.colorbar(cf)
    cf.set_clim(-2, 2)
    # plt.title('(f)$\\sum_{j=1}^{j=4}x_i^j$', fontsize=12)
    # plt.title('(f) Data: Signal + Noise', fontsize=16)
    plt.title('(f) Data: Signal + Noise')
    plt.colorbar(cf)
    plt.ylabel('Space')
    plt.xlabel('Time')
    plt.gca().tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()