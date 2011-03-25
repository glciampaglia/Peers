# coding=utf-8

''' 
Some distance measures from classic goodness of fit (GoF) tests. 

This module contains implementations of GoF test statistics that are not
already part of SciPy. In the case of Chi-squared distance, we provide the
2-samples version, while for Anderson-Darling's A^2 we provide the k-sample
version.

© 2010 G.L. Ciampaglia
'''

from __future__ import division
import numpy as np
from scipy.stats import chisqprob
from rand import auc as c_auc, adk as c_adk
from warnings import warn

__all__ = [ 'auc', 'c_auc', 'cvmt', 'chisq_2sam', 'adk', 'c_adk' ]


#-------------------------------------------------------------------------------
# Area under curve test statistic
#-------------------------------------------------------------------------------

def auc(a,b):
    '''
    computes the Area Under Curve between empirical distribution function of
    datasets a and b
    '''
    na, nb = len(a), len(b)
    a = np.asarray(a)
    b = np.asarray(b)
    a.sort()
    b.sort()
    c = np.hstack([a, b])
    c.sort()
    area = 0.0
    for i in xrange(na+nb-1):
        xi, xip = c[i:i+2]
        xm = (xip - xi) / 2. + xi
        ya = a.searchsorted(xm, 'right') / na
        yb = b.searchsorted(xm, 'right') / nb
        area += (xip - xi) * abs(ya - yb)
    return area

#-------------------------------------------------------------------------------
# Cramér-Von Mises 2-samples criterion (due to Anderson)
#-------------------------------------------------------------------------------

def cvmt(x,y):
    '''
    Calculates the Cramér-Von Mises test statistic for 2 samples. This extension
    of the CVM test is due to Anderson (1962). 
    
    NOTE: this function only computes the test statistic, not the p-value!
    '''
    N = len(x)
    M = len(y)
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.argsort(np.hstack([x,y]))
    r = np.nonzero(( idx >= 0 ) & ( idx < N ))[0] + 1 
    s = np.nonzero(( idx >= N ) & ( idx < N+M))[0] + 1
    i = np.arange(1,N+1)
    j = np.arange(1,M+1)
    U = N * np.sum( ( r - i )**2 ) + M * np.sum( ( s - j )**2 )
    return U / ( N*M*( N + M ) ) - ( 4*M*N - 1 ) / ( 6*( M + N ) )

#-------------------------------------------------------------------------------
# Chi-squared test statistic for 2-samples 
#-------------------------------------------------------------------------------

# FIXME <Thu Nov 25 15:41:19 CET 2010> find an R implementation for testing.
def chisq_2sam(f_obs1, f_obs2):
    """
    Calculates a two-sample chi square test.
       
    The two samples chi square test tests the null hypothesis that the two
    categorical data sample have the same frequencies.
    
    Parameters
    ----------
    f_obs1, f_obs2 : two arrays
        with observed frequencies in each category. The number of categories
        must be the same.
    
    Returns
    -------
    chisquare statistic : float
        The chisquare test statistic
    p : float
        The p-value of the test.
    
    Notes
    -----
    If the number of observation is the same across the two samples, then the
    number of degrees of freedom is equal to the number of bins minus one (due
    to the additional constraint on the sample size), else it is equal to the
    number of bins. The same observations on the size of the sample in
    the one-way chi squared test (see scipy.stats.chisquare) apply also for the
    case with two samples.
    
    Examples
    --------
    >>> chisq2sam(np.ones(10), np.ones(10)) # same frequencies
    (0.0, 1.0)
    >>> chi2, pval = chisq2sam([100,0, 0], [0, 0, 100])
    >>> print chi2
    200.0
    >>> print pval
    2.08848758376e-45
    """
    if len(f_obs1) != len(f_obs2):
        raise ValueError('expecting same number of bins')
    f_obs1, f_obs_2 = np.asarray(f_obs1, dtype=int), np.asarray(f_obs2, dtype=int)
    s1, s2 = np.sum(f_obs1), np.sum(f_obs2)
    if s1 == s2:
        ksntrns = 1 
    else:
        ksntrns = 0
    idx = ( f_obs1 + f_obs2 ) == 0.
    ksntrns += np.sum(idx.astype(int))
    ddof = len(f_obs1) - ksntrns
    ratio1, ratio2 = map(np.sqrt, [ s2 / s1, s1 / s2 ] )
    chisq = (( f_obs1 * ratio1 ) - ( f_obs2 * ratio2 ))**2 / ( f_obs1 + f_obs2 )
    chisq = np.sum(chisq[~idx])
    return chisq, chisqprob(chisq, ddof)

#-------------------------------------------------------------------------------
# Anderson-Darling k-samples test statistic
#-------------------------------------------------------------------------------

# TODO <Thu Nov 25 13:25:54 CET 2010> compute adjusted A in case of ties
def adk(*x, **kwargs):
    '''
    Computes the Anderson-Darling k-sample test statistic. 
    
    Parameters
    ----------
    x - input samples. 
        Each samples should contain at least 4 observations
    std - boolean (default: False)
        if True, return the standardized A_k^2 value, also called t. 
    
    Note: this code does not correct for ties!
    '''
    k = len(x)
    n = map(len, x)
    if any([ n[i] <= 4 for i in xrange(k) ]):
        warn('at least one sample with less than 5 observations',
                category=UserWarning)
    N = np.sum(n)
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
    if kwargs.get('std', False):
        H = np.sum(np.asarray(n, dtype=float)**-1)
        h = np.sum(j**-1)
        g = 0.
        for i in xrange(1,N-1):
            g += np.sum( (j[i:] * (N-i))**-1 )
        a = ( 4*g - 6 ) * ( k - 1 ) + ( 10 - 6*g ) * H
        b = ( 2*g - 4 ) * k**2 + 8*h*k + ( 2*g - 14*h - 4 )*H - 8*h + 4*g - 6
        c = ( 6*h + 2*g - 2 )*k**2 + ( 4*h - 4*g + 6 )*k + ( 2*h - 6 )*H + 4*h
        d = ( 2*h + 6 ) * k**2 - 4*h*k
        V = ( a*N**3 + b*N**2 + c*N + d ) / ( ( N - 1 )*( N - 2 )*( N - 3 ) )
        t = ( A - ( k - 1. ) ) / np.sqrt( V )
        return t
    else:
        return A
