import numpy as np
from center import center
from time import time
from matplotlib import pyplot as plt
#lear all
#general_data
x='hold'
fpcs=center(x)
W=(fpcs.shape[0]+1)/2; #maximum window length
W=65
WW=2*W-1
D=fpcs.shape[1]
NP=D #maximum is D
NFE=W # maximum is W


wt = 'hann'
wt='blackman'
wt='hamming'
wt='bartlett'
#wt='none'
start = time()


fE2,VP,FEP=DAHD4freq_part_weight(fpcs,W,NFE,NP,wt)
elapsed = time()-start
#%% DAHD spectrum
NP=1
plt.figure()
plt.rcParams.update({'font.size': 16})
plt.semilogy(fE2[1:],abs(VP[:,1:]),'ro',markersize=4,markerfacecolor='r')
#hold on
plt.semilogy(fE2[17],abs(VP[:NP,17]),'bo',markersize=8,markerfacecolor='b')
plt.semilogy(fE2[26],abs(VP[:NP,26]),'co',markersize=8,markerfacecolor='c')
plt.semilogy(fE2[46],abs(VP[:NP,46]),'go',markersize=8,markerfacecolor='g')
plt.semilogy(fE2(57),abs(VP[:NP,56]),'ko',markersize=8,markerfacecolor='k')
plt.ylabel('\lambda')
#xlim([0 0.5]);
#ylim([1.e-2 Inf]);
#xlabel('Freq')
#set(gca,'FontSize',16);
#%% compute DAHMs
# EP=zeros((2*W-1)*D,2*NP,NFE);%% 
# X=reshape(fpcs,size(fpcs,1)*D,1);
# tic
#for ifff in range(NFE):

# ER=DAHM4_ex(FEP(:,:,iff),W,iff,2*NP);
# ERR=reshape(ER,(2*W-1)*D,2*NP);
# EP(:,:,iff)=ERR;
# end


