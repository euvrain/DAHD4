import numpy as np

def center(x):
    '''
    CENTER(X) centers the data in X by subtraction of the
    mean XM. If X contains NaNs, indicating missing values, the mean
    of X is computed from the available data.

    See also NANMEAN, MEAN.
    '''
    #error(narginchk(1,1,nargin))           # check number of input arguments 
    if x.ndim > 2:
        raise ValueError("X must be vector or 2-D array.")
    
    #if x is a vector, make sure it is a row vector
    if len(x)==np.prod(x.shape):
        x = x.reshape(-1, 1)
    [m,n] = x.shape

    #get the mean of x
    if np.any(np.any(np.isnan(x))):         # there are missing values in x
        xm = np.nanmean(x,axis=0)
    else:                                   # no missing values
        xm = np.mean(x,axis=0)

    #remove mean
    xc = x - np.tile(xm, (m, 1))