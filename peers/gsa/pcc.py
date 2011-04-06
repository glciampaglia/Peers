'''
Computes sensitivity indices via partial correlation coefficients of each model
parameter on model response and plots scatter plots of each parameter vs
response.

Author: Giovanni Luca Ciampaglia <ciampagg@usi.ch>
'''

from __future__ import division
from argparse import ArgumentParser, FileType
from datetime import datetime
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as pp

from ..utils import sanetext, fmt, rect

# TODO <Fri Mar 25 23:54:07 CET 2011>:
# - make it compute PCC and produce scatter plots for each response variable
#   (right now it assumed only one output variable)
# - use gettxtdata and other functions

# TODO <Wed Feb  2 10:37:39 CET 2011> compute R^2 of regression of y on {x} U z?
# It is computed in gsa_regr.py anyways ...
def pcc(x,y,z):
    '''
    Computes partial correlation coefficient between x and y, given
    variables in z. This is the cosine of the angle between the vectors of
    residuals obtain from regressing separately x and y on the control (or
    confounding) variables in z. The vector of residuals lie in the hyperplane
    orthogonal to z.

    Parameters
    ----------
    x,y - (N,) arrays of variables
    z   - (M,N) array of M control variables

    Returns
    -------
    rho  - the partial correlation coefficient
    t    - the value of the t statistic. Under the null hypothesis that rho = 0
           this is quantity is approximately t-distributed
    pval - the p-value under the null hypothesis
    ddof - number of degrees of freedom

    References
    ----------
    * Wikipedia - Partial Correlation Coefficient - http://en.wikipedia.org/wiki/Partial_correlation
    * From the Penn. State Univ. STAT 505 online course notes - http://www.stat.psu.edu/online/courses/stat505/07_partcor/06_partcor_partial.html 
    '''
    # First find an orthonormal basis for the space spanned by vectors of z.
    # Hat tip to R. Kern and G. Varoquax for the idea of the QR decomposition.
    # See here: http://thread.gmane.org/gmane.comp.python.numeric.general/35633 
    zorth = np.linalg.qr(z.T)[0].T
    # then compute residuals of regression of x respectively y on z 
    x = x - np.dot(np.dot(zorth, x), zorth)
    y = y - np.dot(np.dot(zorth, y), zorth)
    rho = np.dot(x,y)/np.sqrt(np.dot(x,x) * np.dot(y,y))
    n = len(x)
    k = len(z)
    ddof = n - 2 - k
    t = rho * np.sqrt(ddof / (1 - rho ** 2))
    pval = 2*(1 - st.t.cdf(np.abs(t), ddof)) # two-tailed p-value
    return rho, t, pval, ddof

def print_pcc(results):
    '''
    pretty-prints results in a table
    '''
    import subprocess as sp
    try:
        h,w = map(int, sp.Popen('stty size'.split(), stdout=sp.PIPE,
                stderr=sp.PIPE).communicate()[0].strip().split(' '))
    except:
        h,w = 45,85
    print 'Date: %s' % datetime.now()
    print 'Method: partial correlations' 
    print '-'*w
    names = results.keys()
    name_cols_width = max(map(len, names)) + 2
    header = [' '*name_cols_width, 'rho', 't', 'p-value', 'ddof']
    cols_width = int(np.floor((w - name_cols_width) / 4))
    for i in xrange(1,len(header)):
        header[i] = header[i].center(cols_width)
    print ''.join(header)
    print '-'*w
    for name, values in sorted(results.items(), key=lambda k : k[0]):
        rho, t, pval, ddof = values
        row = []
        row.append(name.rjust(name_cols_width))
        for val in values:
            row.append(('%.5g' % val).center(cols_width))
        print ''.join(row)
    print '-'*w

def main(args):
    data = np.loadtxt(args.data, delimiter=args.delimiter)
    fig = pp.figure()
    # set space between plots
    fig.subplotpars.update(wspace=.3,hspace=.5)
    if args.error_bars:
        X = data[:,:-2]
        y = data[:,-2]
        ye = data[:,-1]
        n,d = X.shape
        args.cols, args.rows = rect(d)
        for i,x in enumerate(X.T):
            ax = pp.subplot(args.rows,args.cols,i+1)
            ax.errorbar(x,y,ye/2.0, marker='o', ls='',c='w')
    else:
        X = data[:,:-1]
        y = data[:,-1]
        n,d = X.shape
        args.cols, args.rows = rect(d)
        if d < args.cols:
            args.cols = d
        args.rows = np.ceil(d / args.cols)
        for i,x in enumerate(X.T):
            ax = pp.subplot(args.rows,args.cols,i+1)
            ax.scatter(x,y,c='w')
    # set titles, compute partial correlation coefficients
    pcc_results = {}
    if args.params_file is not None:
        args.params = args.params_file.readline().strip().split(',')
        for i in xrange(d):
            ax = fig.axes[i]
            ax.set_title(sanetext(args.params[i]), fontsize='medium')
            idx = range(d)
            del idx[i]
            pcc_results[args.params[i]] = pcc(X[:,i], y, X[:,idx].T) 
    else:
        for i in xrange(d):
            idx = range(d)
            del idx[i]
            pcc_results['param-%d' % i] = pcc(X[:,i],y,X[:,idx].T)
    print_pcc(pcc_results)
    # set ticks on x/y axis, y label only on first plot
    for i in xrange(d):
        ax = fig.axes[i]
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.linspace(xmin,xmax,3))
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.linspace(ymin,ymax,5))
        if i > 0:
            ax.set_yticklabels([])
        if i == 0:
            ax.set_ylabel(r'$<\tau>$ (days)',fontsize=14)
    if args.output is not None:
        for out in args.output:
            pp.savefig(out, format=fmt(out.name))
    pp.show()

def make_parser():
    parser = ArgumentParser(description='plots scatters of model response vs '\
            'parameter value and computes partial correlation coefficient')
    parser.add_argument('data', type=FileType('r'), metavar='FILE', help='data'\
            ' file')
    parser.add_argument('-p', '--parameters', type=FileType('r'), help='set '\
            'titles to parameters taken from %(metavar)s', dest='params_file', 
            metavar='FILE')
    parser.add_argument('-o', '--output', type=FileType('w'), help='save '\
            'graphics in %(metavar)s', nargs='+', metavar='FILE')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of '\
            'data values', metavar='CHAR')
    parser.add_argument('-e','--error-bars', action='store_true', help='plot '\
            'error bars (standard errors in last column)')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
