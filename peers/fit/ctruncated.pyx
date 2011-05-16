# coding=utf-8
# cython: profile=True

cimport numpy as cnp

import numpy as np
from scipy.cluster.vq import kmeans2
import sys
import cython

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double log(double)
    double M_PI
    double INFINITY

cdef extern double ndtr(double)

@cython.profile(False)
cdef inline double norm_pdf(double x):
    return 1.0/sqrt(2* M_PI)*exp(-x**2/2.0)

# instead of a tuple, these functions get two parameters (l and u) for
# specifying the truncation bounds

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray tnorm_pdf(
        cnp.ndarray[cnp.double_t, ndim=1] x, 
        double mu, 
        double sigma, 
        double l, 
        double u):
    ''' truncated normal density function '''
    cdef int N = len(x)
    cdef double c, xi
    cdef cnp.ndarray[cnp.double_t, ndim=1] d = np.empty((N,))
    u = (u - mu) / sigma
    l = (l - mu) / sigma
    c = ndtr(u) - ndtr(l)
    for i in xrange(N):
        xi = (x[i] - mu) / sigma
        if ((xi >= l) & (xi <= u)):
            d[i] = norm_pdf(xi) / (c * sigma)
        else:
            d[i] = 0.0
    return d

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray tnorm_cdf(
        cnp.ndarray[cnp.double_t, ndim=1] x, 
        double mu, 
        double sigma, 
        double l, 
        double u):
    ''' truncated normal distribution function '''
    cdef int N = len(x)
    cdef double c, xi
    cdef cnp.ndarray[cnp.double_t, ndim=1] p = np.empty((N,))
    u = (u - mu) / sigma
    l = (l - mu) / sigma
    c = ndtr(u) - ndtr(l)
    for i in xrange(N):
        xi = (x[i] - mu) / sigma
        if ((xi >= l) & (xi <= u)):
            p[i] = (ndtr(xi) - ndtr(l)) / c
        else:
            p[i] = 0.0
    return p

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _loglike(
        cnp.ndarray[cnp.double_t, ndim=1] data, 
        cnp.ndarray[cnp.double_t, ndim=1] weights, 
        cnp.ndarray[cnp.double_t, ndim=1] mu, 
        cnp.ndarray[cnp.double_t, ndim=1] sigma,
        double l, 
        double u):
    cdef int n = len(data)
    cdef int k = len(weights)
    cdef cnp.ndarray[cnp.double_t, ndim=1] tmp
    cdef cnp.ndarray[cnp.double_t, ndim=1] accum = np.zeros((n,))
    cdef double ll = 0, w
    for i in xrange(k):
        w = weights[i]
        tmp = tnorm_pdf(data, mu[i], sigma[i], l, u)
        for j in xrange(n):
            accum[j] += tmp[j] * w
    for j in xrange(n):
        ll += log(accum[j])
    return ll

@cython.boundscheck(False)
@cython.wraparound(False)
cdef cnp.ndarray _responsibilities(
        cnp.ndarray[cnp.double_t, ndim=1] data, 
        cnp.ndarray[cnp.double_t, ndim=1] weights, 
        cnp.ndarray[cnp.double_t, ndim=1] mu, 
        cnp.ndarray[cnp.double_t, ndim=1] sigma,
        double l, 
        double u):
    ''' the E-step of the algorithm '''
    cdef int n = len(data)
    cdef int k = len(weights)
    cdef cnp.ndarray[cnp.double_t, ndim=2] g = np.empty((k, n))
    cdef cnp.ndarray[cnp.double_t, ndim=1] tmp
    cdef double wi
    for i in xrange(k):
        tmp = tnorm_pdf(data, mu[i], sigma[i], l, u) 
        wi = weights[i]
        for j in xrange(n):
            g[i,j] = tmp[j] * wi
    g = g.T / g.sum(axis=0)[:,np.newaxis]
    return g

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _tmeancost(double mu, double sigma, double l, double u):
    ''' additive constant for the mean of the truncated normal '''
    cdef double c,d,n
    l = (l - mu) / sigma
    u = (u - mu) / sigma
    n = norm_pdf(u) - norm_pdf(l)
    d = ndtr(u) - ndtr(l)
    c = sigma * n / d
    return c

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _tvarcost(double mu, double sigma, double l, double u):
    ''' multiplicative factor for the variance of the truncated normal '''
    cdef double n_1, n1_1, n1_2, d, c
    l = (l - mu) / sigma
    u = (u - mu) / sigma
    # as x --> +/- Inf, x * f(x) --> 0 for the gaussian density f, but 0. *
    # Inf would normally give NaN
    n1_1 = 0. if u == INFINITY else u * norm_pdf(u) 
    n1_2 = 0. if l == INFINITY else l * norm_pdf(l) 
    n1 = n1_1 - n1_2
    n2 = norm_pdf(u) - norm_pdf(l)
    d = ndtr(u) - ndtr(l)
    c = 1 + n1 / d - (n2 /d )**2 
    assert c > 0, "c = %g " % c
    return c

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _maximize(
        cnp.ndarray[cnp.double_t, ndim=1] data, 
        cnp.ndarray[cnp.double_t, ndim=1] mu, 
        cnp.ndarray[cnp.double_t, ndim=1] sigma, 
        double l, 
        double u, 
        cnp.ndarray[cnp.double_t, ndim=2] gamma):
    ''' the M-step of the algorithm. Moments estimates are for the non-truncated
    normal. '''
    cdef int n = len(data)
    cdef int k = len(sigma)
    cdef cnp.ndarray[cnp.double_t, ndim=1] mu1 = np.zeros((k,))
    cdef cnp.ndarray[cnp.double_t, ndim=1] sigma1 = np.zeros((k,))
    cdef cnp.ndarray[cnp.double_t, ndim=1] w1 = gamma.sum(axis=0) / float(n)
    cdef double accum, norm, mu1i, sigma1i, w1i, muc, sigmac
    for i in xrange(k):
        accum = 0.0
        norm = 0.0
        for j in xrange(n):
            accum += (data[j] * gamma[j,i])
            norm += gamma[j,i]
        mu1i = accum / norm
        accum = 0.0
        for j in xrange(n):
            accum += (data[j] - mu1i) ** 2 * gamma[j,i] 
        sigma1i = sqrt(accum / norm)
        muc = _tmeancost(mu[i], sigma[i], l, u)
        mu1[i] = mu1i - muc
        sigma1[i] = sigma1i / sqrt(_tvarcost(mu[i], sigma[i], l, u))
    return w1, mu1, sigma1

def _init_EM(data, k, prng=np.random):
    ''' initializes with hard assignments to clusters using kmeans '''
    # ensurers deterministic result of kmeans2
    seed = prng.randint(0, sys.maxint)
    np.random.seed(seed)
    flag = True
    n = float(len(data))
    while flag:
        mu, assign = kmeans2(data, k, iter=5, minit='random')
        sigma = []
        weights = []
        for i in xrange(k):
            idx = (assign == i)
            sigma.append(np.std(data[idx], ddof=1))
            weights.append(np.sum(idx) / n)
        sigma = np.asarray(sigma)
        weights = np.asarray(weights)
        flag = True - np.all(weights > 0)
    return weights, mu, sigma

def EM(data, k, bounds=None, n_iter=100, thresh=1e-2, verbose=False, 
        prng=np.random):
    '''
    Fit a truncated GMM to data using the EM algorithm. 
    
    Parameters
    ----------
    data    - array (will be truncated within bounds)
    k       - number of components to fit
    bounds  - default: (data.min(), data.max()) 
    n_iter - maximum number of iterations
    thresh  - stop iteration when marginal increment in loglike is below thresh
    verbose - if True, print log-likelihood and prior probabilities
    prng    - instance of numpy.random.RandomState
    
    Returns
    -------
    weights, mu, sigma - GMM parameters
    loglike            - the sequence of log-likelihood values
    flag               - True if convergence happened within n_iter
    ''' 
    data = np.ravel(data)
    if bounds is not None:
        l, u = bounds
        data = data[(data >= l) & (data <= u)]
    else:
        bounds = (np.min(data), np.max(data))
    l, u = bounds
    weights, mu, sigma = _init_EM(data, k, prng)
    loglike = np.zeros((n_iter,))
    loglike[0] = _loglike(data, weights, mu, sigma, l, u)
    if verbose:
        print "0) LogLike = %g, Priors = %s" % (loglike[0], weights)
    for i in xrange(1, n_iter):
        gamma = _responsibilities(data, weights, mu, sigma, l, u)
        weights, mu, sigma = _maximize(data, mu, sigma, l, u, gamma)
        loglike[i] = _loglike(data, weights, mu, sigma, l, u) 
        if verbose:
            print "%d) LogLike = %g, Priors = %s" % (i, loglike[i], weights)
        flag = np.abs(loglike[i - 1] - loglike[i]) < thresh
        if flag:
            break
    return (weights, mu, sigma, np.trim_zeros(loglike), flag)
