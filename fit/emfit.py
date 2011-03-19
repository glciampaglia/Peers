#!/usr/bin/python
# encoding: utf-8

'''
Estimates parameters of a Gaussian Mixture Model (GMM) from input files and
plots the fit.
'''

from os.path import exists, isdir, join, extsep
import sys
from argparse import ArgumentParser, Action, FileType
import numpy as np
from scikits.learn.mixture import GMM
import matplotlib.pyplot as pp
from scipy.stats import norm

# TODO <Fri Mar 18 14:45:16 CET 2011>:
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

class _CheckDir(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not exists(values) or not isdir(values):
            parser.error('%s is not a directory!' % values)
        setattr(namespace, self.dest, values)

def plot(fn, data, model, bins, **params):
    fig = pp.figure()
    _, edges, _ = pp.hist(data, bins=bins, figure=fig, normed=1, label='data')
    xmin, xmax = edges[0], edges[-1]
    xi = np.atleast_2d(np.linspace(xmin, xmax, 1000)).T
    means = model.means.ravel()
    variances = np.asarray(model.covars).ravel()
    pi = gmmpdf(xi, means, variances, model.weights)
    pp.plot(xi, pi, 'r-', figure=fig, label='GMM fit')
    pp.xlabel(r'$u = \mathrm{log}(\tau)$ (days)', fontsize=16)
    pp.ylabel(r'Probability Density $p(x)$')
    pp.legend(loc='upper left')
    fn = extsep.join(fn.split(extsep)[:-1])
    if 'confidence' in params:
        title = r'%s, $\varepsilon = %g$' % (fn, params['confidence'])
    else:
        title = fn
    pp.title(title)
    pp.draw()
    pp.savefig(fn)
    pp.close()

def main(args):
    if args.names is not None:
        names = args.names.readline().strip().split(args.sep)
    else:
        names = None
    linesiter = ( l.strip() for l in iter(args.index.readline,'') )
    for line in linesiter:
        values, fn = line.split(args.sep)
        if names is not None:
            paramdict = dict(names, values)
        else:
            paramdict = {}
        d = np.load(join(args.directory,fn))
        if args.verbose:
            print fn,
        if len(d):
            gmm = GMM(args.components)
            gmm.fit(d)
            gfn = extsep.join(fn.split(extsep)[:-1]) + extsep + 'pdf'
            plot(gfn, d, gmm, args.bins, **paramdict)
            if args.verbose:
                print 
                sys.stdout.flush()
        elif args.verbose:
            print 'no data!'
            sys.stdout.flush()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('index', type=FileType('r'), help='Data files index.')
    parser.add_argument('components', type=int, help='Number of GMM components.')
    parser.add_argument('-C', '--directory', default='.', help='Interpret '
            'file paths in index as relative to %(metavar)s.', metavar='DIR',
            action=_CheckDir)
    parser.add_argument('-d', '--delimiter', default=',', help='Data files index'
            ' has fields separated by %(metavar)s.', dest='sep', metavar='CHAR')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be '
            'verbose.')
    parser.add_argument('-b', '--bins', type=int, metavar='NUM', help='Number '
            'of histogram bins.', default=10)
    parser.add_argument('-n', '--names', type=FileType('r'), help='Parameter '
            'names file', metavar='FILE')
    ns = parser.parse_args()
    main(ns)
