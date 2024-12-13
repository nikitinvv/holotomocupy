
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage as ndimage
from holotomocupy.chunking import gpu_batch

import cupyx.scipy.ndimage as ndimage
def remove_outliers(data, dezinger, dezinger_threshold):    
    res=cp.array(data)
    if (int(dezinger) > 0):
        w = int(dezinger)
        fdata = ndimage.median_filter(res, [1,w, w])
        print('Affected pixels:',np.sum(np.abs(res-fdata)>fdata*dezinger_threshold))
        res[:] = np.where(np.abs(res-fdata)>fdata*dezinger_threshold, fdata, res)
    return res.get()

@gpu_batch
def linear(x,y,a,b):
    """Linear operation res = ax+by

    Parameters
    ----------
    x,y : ndarray
        Input arrays
    a,b: float
        Input constants    
    
    Returns
    -------
    res : ndarray
        Output array
    """
    return a*x+b*y

@gpu_batch
def _dai_yuan_alpha(d,grad,grad0):
    divident = cp.zeros([d.shape[0]],dtype='float32')
    divisor = cp.zeros([d.shape[0]],dtype=d.dtype)
    for k in range(d.shape[0]):
        divident[k] = cp.linalg.norm(grad[k])**2
        divisor[k] = cp.real(cp.vdot(d[k], grad[k]-grad0[k]))
            
    return [divident,divisor]

def dai_yuan(d,grad,grad0):
    """Dai-Yuan direction for the CG scheme
    
    Parameters
    ----------
    d : ndarray        
        The Dai-Yuan direction from the previous iteration
    grad : ndarray        
        Gradient on the current iteration
    grad0 : ndarray        
        Gradient on the previous iteration    
    
    Returns
    -------
    res : ndarray
        New Dai-Yuan direction
    """
    
    [divident,divisor] = _dai_yuan_alpha(d,grad,grad0)    
    alpha = np.sum(divident)/np.sum(divisor)
    res = linear(grad,d,-1,alpha)
    return res 
