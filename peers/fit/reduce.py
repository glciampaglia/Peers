#!/usr/bin/env python

'''Reduction script for Gaussian Mixture Model (GMM) parameter estimation. '''

import sys
import os.path
from warnings import warn
from itertools import groupby
from argparse import ArgumentParser, FileType 
import numpy as np
import matplotlib.pyplot as pp
from scipy.stats import norm
from scikits.learn.mixture import GMM
import csv

from .truncated import TGMM, plot as plottruncated
from ..graphics import mixturehist
from ..utils import fmt, sanetext

_fields = [ 'mean-%d', 'variance-%d', 'weight-%d' ] 

def _getfields(n):
    global _fields
    return reduce(list.__add__, [[ f % i for f in _fields ] for i in xrange(n)])

def _reduce(args):
    reader = csv.reader(args.indexfile, delimiter=',', quoting=0)
    params = reader.next()
    N = len(params) - 1
    index = dict([ (row[-1], map(float, row[:-1])) for row in reader ])
    reader = csv.reader(args.datafile, delimiter=',', quoting=0)
    for row in reader:
        key = row[0]
        value = map(float, row[1:])
        if key in index:
            index[key] += value
        else:
            raise ValueError('not in index: %s' % key)
    arr = np.asarray(index.values())
    idx = arr[:,:N].argsort(axis=0)[:,0]
    arr = arr[idx]
    C = arr.shape[1] - N
    fields = params[:-1] + _getfields(C/3)
    writer = csv.DictWriter(sys.stdout, fields, delimiter=',', quoting=0)
    kf = lambda k : tuple(k[:N])
    writer.writeheader()
    for key, subiter in groupby(arr, kf):
        row = dict(zip(fields, key))
        x = np.asarray([ r[N:] for r in subiter ])
        row.update(zip(fields[N:], x.mean(axis=0)))
        writer.writerow(row)

def _fit(args):
    data = np.load(args.datafile)
    if args.log:
        data = np.log(data)
    if len(data) == 0:
        raise ValueError('empty data file: %s' % args.datafile)
    if args.truncated:
        model = TGMM(args.components)
    else:
        model = GMM(args.components)
    model.fit(data, n_iter=args.iterations)
    means = model.means.ravel()
    sigmas = np.sqrt(model.covars).ravel()
    weights = model.weights.ravel()
    idx = means.argsort()
    beta = [means[idx], sigmas[idx], weights[idx]]
    beta = (args.datafile,) + reduce(tuple.__add__, map(tuple, beta))
    print ','.join(map(str,beta))

#def plot(data, model, bins=10, output=None, **params):
#    '''
#    plots histogram of log-lifetime data with GMM fit (as stacked densities) and
#    save output to file 
#    '''
#    fig = pp.figure()
#    means = model.means.ravel()
#    variances = np.asarray(model.covars).ravel()
#    RV = map(norm, means, variances)
#    mixturehist(data, RV, model.weights, figure=fig)
#    pp.xlabel(r'$u = \mathrm{log}(\tau)$ (days)')
#    pp.ylabel(r'Prob. Density $p(x)$')
#    if len(params):
#        title = ', '.join(map(lambda k : ' = '.join(k), params.items()))
#        pp.title(sanetext(title), fontsize='small')
#    elif output is not None:
#        pp.title(sanetext(output.name), fontsize='small')
#    pp.draw()
#    if output is not None:
#        pp.savefig(output, format=fmt(output, 'pdf'))
#    pp.show()

def main(args):
    args.func(args)

def make_parser():
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers()
    fitparser = subparsers.add_parser('fit', help='mixture parameters estimate')
    fitparser.add_argument('datafile', metavar='data', help='NumPy binary array file')
    fitparser.add_argument('-c', '--components', type=int, default=2, 
            help='default: %(default)d')
    fitparser.add_argument('-t', '--truncated', action='store_true')
    fitparser.add_argument('-i', '--iterations', type=int, default=1000, 
            help='EM iterations (default: %(default)d')
    fitparser.add_argument('-l', '--log', action='store_true', help='use log-data')
    fitparser.set_defaults(func=_fit)
    redparser = subparsers.add_parser('reduce', help='reduction by averaging')
    redparser.add_argument('indexfile', metavar='index', type=FileType('r'))
    redparser.add_argument('datafile', metavar='data', help='fit results',
            type=FileType('r'))
    redparser.set_defaults(func=_reduce)
#    parser.add_argument('-p', '--plot', action='store_true', 
#            help='stacked area plot')
#    parser.add_argument('-b', '--bins', type=int, metavar='NUM', default=10)
#    parser.add_argument('-f' ,'--format', help='graphic output format (default:'
#            ' %(default)s)', default='pdf')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
