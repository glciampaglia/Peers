#coding: utf-8
#cython: profile=True
from __future__ import division
import numpy as np
from collections import deque
from warnings import warn

cimport numpy as cnp
cimport cython

@cython.profile(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int _binsearch(double x, object seq, int start, int stop):
    '''
    See binsearch
    '''
    cdef cnp.ndarray[cnp.float64_t, ndim=1] _seq = seq
    if x < _seq[start]:
        return 0
    if x >= _seq[stop]:
        return len(seq)
    if start == stop - 1:
        return stop
    cdef int mid = <int>((stop - start) / 2) + start
    if x == _seq[mid]:
        return mid
    elif x < _seq[mid]:
        return _binsearch(x, _seq, start, mid)
    else:
        return _binsearch(x, _seq, mid, stop)

def binsearch(x, seq):
    ''' 
    Binary search. 
    
    The array seq *must* be ordered and must not contain any duplicate.
    
    Returns the index into seq such that inserting before the index would keep
    seq sorted. If all(x < seq) returns 0, if all(x > seq) returns N
    '''
    seq = np.asarray(seq, dtype=np.double)
    return _binsearch(float(x), seq, 0, len(seq)-1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.profile(False)
cdef inline object _randwpmf(object pmf, int num, object prng):
    '''
    See doctype of _randwpmf.randwpmf
    '''
    if num < 0:
        raise ValueError('invalid number of elements: %s' % num)
    cdef:
        cnp.ndarray[cnp.double_t, ndim=1] _pmf = np.asarray(pmf, dtype=np.double)
        int i, n = len(_pmf)
        cnp.ndarray[cnp.double_t, ndim=1] cdf = np.empty([ n ], dtype=np.double)
        cnp.double_t norm_sum = 0.0
        cnp.ndarray[cnp.double_t, ndim=1] u = prng.random_sample(num)
    norm_sum = 0.
    for i in xrange(n):
        norm_sum += _pmf[i]
    if norm_sum <= 0.:
        raise ValueError('argument is not a valid probability mass function')
    for i in xrange(n):
        cdf[i] = _pmf[i] / norm_sum
        if i > 0:
            cdf[i] = cdf[i-1] + cdf[i]
    res = deque()
    for j in xrange(len(u)):
        res.append(binsearch(u[j], cdf, 0, n-1))
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

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef cnp.float64_t adk(object x):
    cdef int k, N, I, nmi 
    cdef double Nd, kd, incsum, A, H, h, g, a, b, c, d, V, t    # 
    cdef int i, ii, ji                                          # indices
    cdef cnp.ndarray[cnp.int_t, ndim=1] r, idx, n               
    cdef cnp.ndarray[cnp.double_t, ndim=1] C, M, j, den, num
    k = len(x)
    kd = <double>k
    n = np.asarray(map(len, x))
    if np.any( n <= 4 ):
        warn('at least one sample with less than 5 observations',
                category=UserWarning)
    N = np.sum(n)
    Nd = <double>N
    x = map(np.asarray, x)
    idx = np.argsort(np.hstack(x)) # sorting indices
    I = 0
    j = np.arange(1,N, dtype=float) # [1 ... N-1]
    den = j * j[::-1]       # inner summand's denominator
    incsum = 0.             # incremental summand
    for i in xrange(k):
        C = np.zeros((N,))
        # ranks of observations from i-th sample in the full sample
        r = np.nonzero((idx >= I) & (idx < I+n[i]))[0]
        # C is an indicator array for sample i: C[:j].sum() = M_ij
        C[r] = 1 
        M = C.cumsum()
        # inner summation goes from 1 to N-1
        num = (N * M[:-1] - j * n[i])**2 
        incsum += np.sum(num/den) / n[i]
        I += n[i]
    A = incsum / N
    return A
#    H = np.sum(np.asarray(n, dtype=float)**-1)
#    h = np.sum(j**-1)
#    g = 0.
#    for ii in xrange(1,N-1):
#        nmi = N - ii
#        for ji in xrange(ii, N-1):
#            g += 1. / ( (ji+1) * nmi )
#    a = ( 4*g - 6 ) * ( kd - 1 ) + ( 10 - 6*g ) * H
#    b = ( 2*g - 4 ) * kd**2 + 8*h*kd + ( 2*g - 14*h - 4 )*H - 8*h + 4*g - 6
#    c = ( 6*h + 2*g - 2 )*kd**2 + ( 4*h - 4*g + 6 )*kd + ( 2*h - 6 )*H + 4*h
#    d = ( 2*h + 6 ) * kd**2 - 4*h*kd
#    V = ( a*Nd**3 + b*Nd**2 + c*Nd + d ) / ( ( Nd - 1 )*( Nd - 2 )*( Nd - 3 ) )
#    t = ( A - ( kd - 1. ) ) / np.sqrt( V )
##    print A, H, h, g, a, b, c, d, V, t
#    return t

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef auc(object a, object b):
    '''
    computes the Area Under Curve between empirical distribution function of
    datasets a and b
    '''
    cdef int na, nb, i
    cdef cnp.ndarray[cnp.double_t, ndim=1] _a,_b, c
    cdef double area, xi, xip, xm, ya, yb
    na, nb = len(a), len(b)
    _a= np.asarray(a)
    _b = np.asarray(b)
    a.sort()
    b.sort()
    c = np.hstack([a, b])
    c.sort()
    area = 0.0
    for i in xrange(na+nb-1):
        xi = c[i]
        xip = c[i+1]
        xm = (xip - xi) / 2. + xi
        ya = _binsearch(xm, a, 0, na-1) / na
        yb = _binsearch(xm, b, 0, nb-1) / nb
        area += (xip - xi) * abs(ya - yb)
    return area
