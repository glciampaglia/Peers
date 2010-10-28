#coding: utf-8
#cython: profile=True
from __future__ import division
import numpy as np
from collections import deque

cimport numpy as cnp
cimport cython


DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

@cython.profile(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int _supidx(float x, object seq, int start, int stop):
    ''' binary search: recursively find the index of sup{x} in seq (i.e. the
    element on the right). The sequence *must* be ordered '''
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _seq = seq
    if x <= _seq[start]:
        return start
    if stop - start == 1:
        return stop
    cdef int mid = <int>((stop - start) / 2) + start
    if x == _seq[mid]:
        return mid
    elif x < _seq[mid]:
        return _supidx(x,_seq,start,mid)
    else:
        return _supidx(x,_seq,mid,stop)

def supidx(x,seq,start,stop):
    return _supidx(x,np.asarray(seq, dtype=DTYPE),start,stop)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cdef inline object _randwpmf(object pmfarr, int num, object prng):
    '''
    See doctype of _randwpmf.randwpmf
    '''
    if num < 0:
        raise ValueError('invalid number of elements: %s' % num)
    cdef:
        cnp.ndarray[DTYPE_t, ndim=1] pmf = np.asarray(pmfarr, dtype=DTYPE)
        int i, n = len(pmf)
        cnp.ndarray[DTYPE_t, ndim=1] cdf = np.empty([ n ], dtype=DTYPE)
        DTYPE_t norm_sum = 0.0
        cnp.ndarray[DTYPE_t, ndim=1] u = prng.random_sample(num)
    norm_sum = 0.
    for i in xrange(n):
        norm_sum += pmf[i]
    if norm_sum <= 0.:
        raise ValueError('argument is not a valid probability mass function')
    for i in xrange(n):
        cdf[i] = pmf[i] / norm_sum
        if i > 0:
            cdf[i] = cdf[i-1] + cdf[i]
    res = deque()
    for j in xrange(len(u)):
        res.append(_supidx(u[j], cdf, 0, n))
    return np.asarray(res)

cpdef object randwpmf(object pmf, int num=1, object prng=np.random):
    '''
    samples an array of random integers with prescribed probability mass
    function 'pmf'. The size of the array is given, else a random scalar is
    returned.

    Parameters
    ----------
    pmf     - a sequence of probability masses (e.g. bin frequencies)
    num     - number of random variates (1 returns a scalar, otherwise an array)
    prng    - numpy.random.RandomState object (default = numpy.random)
    '''
    return _randwpmf(pmf, num, prng)
