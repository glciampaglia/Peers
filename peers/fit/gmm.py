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

from ..utils import CheckDirAction, sanetext, fmt

# TODO <Tue Apr  5 11:03:34 CEST 2011>:
# - plot density of individual components and not of mixture
# - add subplot like in R's histogram function with vertical lines for each
#   observation. Colors of lines are mapped on the a scale red to blue (or
#   whatever color is chosen for the density lines of the components.

def gmmpdf(x, means, variances, weights):
    res = []
    x = x.ravel()
    for m,v in zip(means, variances):
        res.append(norm.pdf(x, m, np.sqrt(v)))
    return np.sum(np.asarray(res) * weights[:,None], axis=0)

def plot(fn, data, model, bins, **params):
    fig = pp.figure()
    _, edges, _ = pp.hist(data, bins=bins, figure=fig, normed=1, label='data',
            fc='w')
    xmin, xmax = edges[0], edges[-1]
    xi = np.atleast_2d(np.linspace(xmin, xmax, 1000)).T
    means = model.means.ravel()
    variances = np.asarray(model.covars).ravel()
    pi = gmmpdf(xi, means, variances, model.weights)
    pp.plot(xi, pi, 'r-', figure=fig, label='GMM fit')
    pp.xlabel(r'$u = \mathrm{log}(\tau)$ (days)', fontsize=16)
    pp.ylabel(r'Probability Density $p(x)$',fontsize=16)
    pp.legend(loc='upper left')
    if 'confidence' in params:
        c = sanetext(params['confidence'])
        title = r'%s, $\varepsilon = %s$' % (sanetext(fn), c)
    else:
        title = sanetxt(fn)
    pp.title(title)
    pp.draw()
    pp.savefig(fn, format=fmt(fn, 'pdf'))
    pp.close()

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
                gmm = GMM(args.components)
                gmm.fit(d)
                mu, si, we = map(np.ravel, 
                        [gmm.means, np.asarray(gmm.covars), gmm.weights])
                idx = mu.argsort()
                if args.plot:
                    gfn = os.path.splitext(fn)[0] + os.path.extsep + args.format
                    plot(gfn, d, gmm, args.bins, **paramsdict)
                beta.append(np.hstack([mu[idx], si[idx], we[idx]]))
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
