import numpy as np
from scipy.signal import lfilter
def dahc(X,E):
    '''
 Syntax: dahc(X,E);
 This function calculates the 'DAH coefficients' of the data
 matrix X (N x L) from the eigenfunction matrix E (from DAHD function).
 The eigenvector matrix may be of reduced size (k columns) - one
 DAHC will be calculated for each supplied eigenvector.

 Output: A - DAH coeffcients matrix (N-M+1 x k)

  See Appendix A of Chekroun, M. D., and D. Kondrashov, 2017:
  Data-adaptive harmonic spectra and multilayer Stuart-Landau models,
  Chaos, 27, 093110: doi:10.1063/1.4989400
  Written by Dmitri Kondrashov.   Version date 1/13/18
  Please send comments and suggestions to dkondras@atmos.ucla.edu
'''

    [N,L]=X.shape
    [M,k]=E.shape
    M=M//L # Calculate the *actual* value for M
    if k>L*M:
        raise ValueError('Improper specification of k.')
    A=np.zeros((N,k))
    a=np.zeros((N,L))
    Ej=np.zeros((M,L))

    for K in range(k):
        Ej = E[:,K].reshape(M, L)
        Ej = np.flipud(Ej)
        for j in range(L):
            a[:,j] = lfilter(Ej[:,j],1,X[:,j])
        if L>1:
            A[:,K]=np.sum(a,axis=1)
        else:
            A[:,K]=a
        
        
    A=A[M-1:,:]


