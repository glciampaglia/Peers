#!/usr/bin/env python

# encoding: utf-8
# author: Giovanni Luca Ciampaglia <ciampagg@usi.ch>

''' Fits data to a sigmoid function by least squares. '''

from __future__ import division
import re
import numpy as np
from scipy.optimize import leastsq
from scipy.stats import chisqprob, kstest
import matplotlib.pyplot as pp
from argparse import ArgumentParser, FileType
from datetime import datetime

def ttysize():
    import subprocess as sp
    try:
        size = sp.Popen('stty size'.split(), stdout=sp.PIPE, 
                stderr=sp.PIPE).communicate()[0].strip()
        return map(int, size.split(' '))
    except:
        return 45,85

def fmt(f):
    return re.search('\.(\w+)$', f.name).group()[1:]

def sigm(t,a,b,l,c):
    '''
    General sigmoid function with location (l), scale (b) and saturation (a)
    parameter and shift (c) parameter. Because of the location and scale, this
    is a non-linear model in both t and the parameters.
    '''
    t = (t - l) / b
    return a / (1 + np.exp(-t)) + c

def resid(p, y, t, s=None):
    a,b,l,c = p
    if s is None:
        return y - sigm(t,a,b,l,c)
    else:
        return (y - sigm(t,a,b,l,c)) / s

# def sigm_der(t,a,b,l,c):
#     '''
#     First derivative w.r.t. to the parameters
#     '''
#     t = (t - l) / b
#     dera = 1 / (1 + np.exp(-t))
#     derb = - (a * (t / b) * np.exp(-t)) / (1 + np.exp(-t)) ** 2
#     derl = (-(a / b) * np.exp(-t)) / (1 + np.exp(-t)) ** 2
#     derc = np.ones(len(t))
#     return np.asarray([dera, derb, derl, derc]).T
# 
# def resid_der(p, y, t):
#     a,b,l,c = p
#     return - sigm_der(t,a,b,l,c)

def fit(x,y,s=None):
    '''
    Fits a sigmoid by least square or by Chi-squared fitting

    Parameters
    ----------
    x, y - IID data observations
    s    - standard deviation of each y

    If s is not None, then a Chi-squared fitting is performed and the standard
    deviation of the data is estimated. 
    
    Minimization is performed with scipy.optimization.leastsq, which is a
    wrapper the MINPACK's implementation of the Levenberg-Marquardt algorithm.
    '''
    xmax, xmin = max(x), min(x)
    x0 = ((xmax - xmin) / 2, (xmax - xmin) / 2, xmax / 2, xmax / 2)
    if s is not None:
        args = (y,x,s)
    else:
        args = (y,x)
    p, cov_x, infodict, mesg, ier = leastsq(resid, x0, args=args, 
            full_output=True, warning=True)
    N = len(x)
    P = len(p)
    fa,fb,fl,fc = p
    ea,eb,el,ec = np.sqrt(np.diag(cov_x))
    _h, _w = ttysize()
    print '-' * _w
    print 'General logistic fit'
    print '-' * _w
    print 'Date: %s' % datetime.now()
    print
    print 'Coefficients'
    print '------------'
    print 'a = %g +/- %g' % (fa, ea)
    print 'b = %g +/- %g' % (fb, eb)
    print 'l = %g +/- %g' % (fl, el)
    print 'c = %g +/- %g' % (fc, ec)
    print
    print 'Goodness-of-fit'
    print '---------------'
    # fvec is a vector of function evaluations at the solution of the least
    # square problem
    fvec = infodict['fvec']
    # residuals of the fit
    r = y - sigm(x, *p)
    if s is not None:
        # compute goodness-of-fit Chi-squared and significance level
        ddof = N - P
        chisq = np.sum(fvec ** 2)
        chisqpval = chisqprob(chisq, ddof)
        print 'Chi-squared: %g, p-value: %g, df: %d' % (chisq, chisqpval, ddof),
        if chisqpval < .05:
            print 'Result: NOT significant'
        else:
            print 'Result: Significant'
    # compute coefficient of determination
    ym = y.mean()
    Rsq = 1 - np.sum(r ** 2)/ np.sum((y - ym) ** 2)
    adjRsq = 1 - (1 - Rsq) * (N -1) / (N - P - 1)
    print 'R-squared: %g, Adjusted R-squared: %g' % (Rsq, adjRsq)
    print
    print 'Analysis of residuals'
    print '---------------------'
    ks, kspval = kstest((r - r.mean())/r.std(), 'norm')
    print 'K-S (residuals): %g, p-value: %g' % (ks, kspval),
    if kspval < .05:
        print 'Result: Reject normality (FAIL)'
    else:
        print 'Result: Cannot reject normality (PASS)'
    return p, r

def test_data(N=50, xmin=-5, xmax=5, s=.1):
    ''' samples noisy data with random parameters from a sigmoid function '''
    a = np.random.uniform(0, xmax / 2)
    b = np.random.uniform(0, xmax / 2)
    l = np.random.uniform(xmin/2, xmax/2)
    c = np.random.uniform(0, xmax / 2)
    t = np.linspace(xmin, xmax, N)
    e = np.random.normal(size=t.shape) * s
    y = sigm(t, a, b, l, c) + e
    return t, y, s * np.ones(len(t)), np.asarray([a, b, l, c])

def plot_fit(x,y,r,p,truep=None,output=None):
    '''
    Parameters
    ----------
    x,y - observations
    r   - residuals
    p   - fitted coefficients
    tp  - true coefficients (optional)
    '''
    #
    # main plot
    #
    N = len(x)
    xmin, xmax = min(x), max(x)
    xi = np.linspace(xmin, xmax, N * 2)
    yi = sigm(xi, *p)
    pp.plot(x,y,'o',ls='', c='k', label='observed')
    pp.plot(xi, yi, 'r-', label='fit')
    if truep is not None:
        pp.plot(xi, sigm(xi, *truep), 'b--', label='actual')
    pp.axis('tight')
    pp.xlabel(r'confidence $\varepsilon$', fontsize=16)
    pp.ylabel(r'$u = \mathrm{ln}(\tau)$ (log-days)', fontsize=16)
    pp.legend(loc='upper left')
    ymin,ymax = pp.ylim()
    ylen = (ymax - ymin)
    pp.ylim(ymin - .1 * ylen, ymax + .1 * ylen)
    #
    # inset with histogram of residuals
    #
    a = pp.axes([.6, .2, .25, .25])
    pp.hist(r, axes=a, bins=10, fc='w')
    rm, rM = pp.xlim()
    ym, yM = pp.ylim()
    pp.setp(a, xticks=[rm, 0, rM], yticks=np.linspace(ym,yM,3))
    pp.setp(a, ylabel='freq.')
    pp.title('residuals')
    #
    pp.draw()
    if output is not None:
        pp.savefig(output, format=fmt(output))
    pp.show()

def main(args):
    if ns.input is not None:
        data = np.loadtxt(ns.input, delimiter=ns.delimiter).T
        if ns.use_errors:
            x, y, s = data[:3]
            p, r = fit(x, y, s)
        else:
            x, y = data[:2]
            p, r = fit(x, y)
        plot_fit(x, y, r, p, output=ns.output)
    else:
        x, y, s, tp = test_data()
        p, r = fit(x, y, s)
        plot_fit(x, y, r, p, tp, output=ns.output)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
            epilog='With no option, fits data sampled from a random sigmoid.')
    parser.add_argument('-i', '--input', type=FileType('r'), help='input data',
            metavar='FILE')
    parser.add_argument('-d', '--delimiter', default=',', metavar='CHAR',
            help='input fields separator (default: "%(default)s")')
    parser.add_argument('-e', '--use-errors', action='store_true', help='third'
            ' input data field contains data standard deviations.')
    parser.add_argument('-o', '--output', type=FileType('w'), help='save figure'
            ' to %(metavar)s', metavar='FILE')
    ns = parser.parse_args()
    main(ns)

