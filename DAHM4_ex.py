import numpy as np

def DAHM4_ex(vv,W,NF,NT):
#  this function computes EVP - space-time DAHMs obtained 
#  by an inverse Fourier transform of Hermitian DAHD eigenvectors vv
#  W -  embedding window
#  NF - frequency index 
#  NT - number of DAHMs to compute
# %
#  See Kondrashov D et al. 2020 Data-adaptive harmonic analysis of oceanic waves and turbulent flows. 
#  Chaos, 30, doi: 10.1063/5.0012077 
#  and 
#  Kondrashov D., et al. 2026 Accurate and robust real-time prediction of September Arctic sea ice 
#  Chaos: 36, doi: 10.1063/5.0295634
#  Written by Dmitri Kondrashov.   Version date 6/22/26
#  Please send comments and suggestions to dkondras@atmos.ucla.edu           
    D=vv.shape[0]

    EVP=np.zeros((2*W-1)*D,NT)

    fftv=np.zeros((2*W-1,D,NT))
    if NF>1:
        for i in range(NT/2):
            i2=2*i-1
            cvv=np.sqrt(-1)*vv[:,i]
            fftv[NF,:,i2]=np.sqrt(W-0.5)*vv[:,i]
            fftv[end-NF+2,:,i2]=np.sqrt(W-0.5)*np.conj(vv[:,i])
            fftv[NF,:,i2+1]=np.sqrt(W-0.5)*cvv
            fftv[end-NF+2,:,i2+1]=np.sqrt(W-0.5)*np.conj(cvv)
        else:
            for i in range(NT/2):
                cvv=np.sqrt(-1)*vv[:,i]
                fftv[NF,:,i]=np.sqrt(2*W-1)*vv[:,i]
        Evv=ifft(fftv,axis=0)
        EVP=np.reshape(Evv,Evv.shape[0]*Evv.shape[1],Evv.shape[2]) 
         
        return