#!/usr/bin/env python

'''Reduction script for Gaussian Mixture Model (GMM) parameter estimation.'''

import sys
import os.path
from warnings import warn
from itertools import groupby
from argparse import ArgumentParser, FileType 
import numpy as np
from scikits.learn.mixture import GMM
from scipy.stats import norm
import matplotlib.pyplot as pp
from matplotlib import cm

from .truncated import TGMM
from ..utils import CheckDirAction, sanetext, fmt

#TODO: move this function in a separate graphics.py file in the top-level package
def plot(data, model, bins=10, output=None, **params):
    '''
    produces stacked area plots of density of a GMM
    '''
    global cm
    fig = pp.figure()
    # transparent histogram
    _, edges, _ = pp.hist(data, bins=bins, figure=fig, normed=1, fc=(0,0,0,0), 
            ec='k')
    xmin, xmax = pp.xlim()
    xi = np.linspace(xmin, xmax, 1000)
    means = model.means.ravel()
    variances = np.asarray(model.covars).ravel()
    RV = map(norm, means, variances)
    pi = [ w * rv.pdf(xi) for rv, w in zip(RV, model.weights) ]
    pi = [ np.zeros(len(xi)) ] + pi
    pi = np.cumsum(pi, axis=0)
    # this colormapping should be photocopy-able
    colors = cm.YlGnBu(np.linspace(0,1,len(pi)-1)*(1- 1.0/len(pi))) 
    for i in xrange(1,len(pi)):
        pp.fill_between(xi, pi[i-1], pi[i], color=colors[i-1])
    pp.xlabel(r'$u = \mathrm{log}(\tau)$ (days)')
    pp.ylabel(r'Prob. Density $p(x)$')
    if 'confidence' in params:
        c = sanetext(params['confidence'])
        title = r'$\varepsilon = %g$' % float(c)
        pp.title(title, fontsize='small')
    elif output is not None:
        title = sanetext(output.name)
        pp.title(title, fontsize='small')
    pp.draw()
    if output is not None:
        pp.savefig(output, format=fmt(output.name, 'pdf'))
    pp.show()

def main(args):
    sep = args.sep
    if args.names is not None:
        names = args.names.readline().strip().split(sep)
    else:
        names = None
    liter = (tuple(l.strip().split(sep)) for l in iter(args.index.readline,''))
    for values, subiter in groupby(liter, lambda k : k[:-1]):
        if names is not None:
            paramsdict = dict(zip(names, values))
        else:
            paramsdict = {}
        beta = []
        for fn in ( s[-1] for s in subiter ):
            d = np.load(os.path.join(args.directory, fn))
            if args.log:
                d = np.log(d)
            if len(d):
                if args.truncated:
                    model = TGMM(args.components)
                else:
                    model = GMM(args.components)
                model.fit(d, n_iter=args.iterations)
                means = model.means.ravel()
                sigmas = np.sqrt(model.covars).ravel()
                weights = model.weights.ravel()
                idx = means.argsort()
                b = np.hstack([means[idx], sigmas[idx], weights[idx]])
                beta.append(b)
                if args.plot:
                    fn1, ext = os.path.splitext(fn)
                    fn1 += '.' + args.format
                    plot(d, model, args.bins, output=fn1, **paramsdict)
            else:
                warn('fn has no data. Skipping.', category=UserWarning)
        if len(beta) == 0:
            warn('Skipping parameters vector: %s.' % '\n\t'.join(values),
                    category=UserWarning)
            continue
        beta = np.asarray(beta).mean(axis=0)
        out = values + tuple(beta)
        print sep.join(['%s'] * len(out)) % out

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('index', type=FileType('r'), help='Data files index.')
    parser.add_argument('components', type=int, help='Number of GMM components.')
    parser.add_argument('-C', '--directory', default='.', help='Interpret '
            'file paths in index as relative to %(metavar)s.', metavar='DIR',
            action=CheckDirAction)
    parser.add_argument('-p', '--plot', action='store_true', help='plot ' 
            'histograms of data with fit')
    parser.add_argument('-t', '--truncated', action='store_true')
    parser.add_argument('-i', '--iterations', type=int, default=100)
    parser.add_argument('-l', '--log', action='store_true', help='Take logs of '
            'data')
    parser.add_argument('-d', '--delimiter', default=',', help='Data files index'
            ' has fields separated by %(metavar)s.', dest='sep', metavar='CHAR')
    parser.add_argument('-b', '--bins', type=int, metavar='NUM', help='Number '
            'of histogram bins to plot', default=10)
    parser.add_argument('-n', '--names', type=FileType('r'), help='Parameter '
            'names file', metavar='FILE')
    parser.add_argument('-f' ,'--format', help='graphic output format (default:'
            ' %(default)s)', default='pdf')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
