#coding: utf-8
#cython: profile=True
from __future__ import division
import numpy as np
cimport numpy as cnp
cimport cython

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cdef inline int _randwpmf(object pmfarr, object prng=np.random) except? -1:
    '''
    Scalar function. For more details, see doctype of _randwpmf.randwpmf
    '''
    cdef:
        cnp.ndarray[DTYPE_t, ndim=1] pmf = pmfarr
        int n = <int> len(pmf)
        cnp.ndarray[DTYPE_t, ndim=1] cdf = np.empty([ n ], dtype=DTYPE)
        DTYPE_t norm_sum = 0.0
        DTYPE_t u = prng.random_sample()
        int i
    norm_sum = 0.
    for i in xrange(len(pmf)):
        norm_sum += pmf[i]
    if norm_sum <= 0.:
        raise ValueError('argument is not a valid probability mass function')
    for i in range(n):
        cdf[i] = pmf[i] / norm_sum
    for i in range(n):
        if i > 0:
            cdf[i] = cdf[i-1] + ( pmf[i] / norm_sum )
        if cdf[i] >= u:
            return i

cpdef randwpmf(pmf, size=None, prng=np.random):
    '''
    samples an array of random integers with prescribed probability mass
    function 'pmf'. The size of the array is given, else a random scalar is
    returned.

    Parameters
    ----------
    pmf     - a numpy array
    size    - size of output array (optional)
    prng    - numpy.random.RandomState object (default = numpy.random)
    '''
    pmf = np.asarray(pmf, dtype=DTYPE)
    if size is not None:
        numel = np.prod(size)
        x = np.asarray([ _randwpmf(pmf,prng) for i in xrange(numel) ])
        return x.reshape(size)
    else:
        return _randwpmf(pmf, prng)

