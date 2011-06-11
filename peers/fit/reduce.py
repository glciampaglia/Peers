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

def main(args):
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(args.index.read(1000))
    args.index.seek(0)
    reader = csv.DictReader(args.index, dialect=dialect)
    keyfun = lambda k : tuple([ k[var] for var in reader.fieldnames[:-1] ])
    data_archive = np.load(args.data)
    for values, subiter in groupby(reader, keyfun):
        beta = []
        for row in subiter:
            k, _ = os.path.splitext(row['file'])
            d = data_archive[k]
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
                    fn, ext = os.path.splitext(row['file'])
                    fn += '.' + args.format
                    if args.truncated:
                        plottruncated(d, model, args.bins, output=fn)
                    else:
                        plot(d, model, args.bins, output=fn, **params)
            else:
                warn('no data: %s' % row['file'], category=UserWarning)
        if len(beta) == 0:
            warn('skipping: %s.' % ', '.join(values), category=UserWarning)
            continue
        beta = np.asarray(beta).mean(axis=0)
        out = values + tuple(beta)
        print ','.join(['%s'] * len(out)) % out
        sys.stdout.flush()

def plot(data, model, bins=10, output=None, **params):
    '''
    plots histogram of log-lifetime data with GMM fit (as stacked densities) and
    save output to file 
    '''
    fig = pp.figure()
    means = model.means.ravel()
    variances = np.asarray(model.covars).ravel()
    RV = map(norm, means, variances)
    mixturehist(data, RV, model.weights, figure=fig)
    pp.xlabel(r'$u = \mathrm{log}(\tau)$ (days)')
    pp.ylabel(r'Prob. Density $p(x)$')
    if len(params):
        title = ', '.join(map(lambda k : ' = '.join(k), params.items()))
        pp.title(sanetext(title), fontsize='small')
    elif output is not None:
        pp.title(sanetext(output.name), fontsize='small')
    pp.draw()
    if output is not None:
        pp.savefig(output, format=fmt(output, 'pdf'))
    pp.show()

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('index', type=FileType('r'), help='index file')
    parser.add_argument('data', help='data file (NPZ)', type=FileType('r'))
    parser.add_argument('-c', '--components', type=int, default=2)
    parser.add_argument('-p', '--plot', action='store_true', 
            help='stacked area plot')
    parser.add_argument('-t', '--truncated', action='store_true')
    parser.add_argument('-i', '--iterations', type=int, default=100)
    parser.add_argument('-l', '--log', action='store_true', help='fit log-data')
    parser.add_argument('-b', '--bins', type=int, metavar='NUM', default=10)
    parser.add_argument('-f' ,'--format', help='graphic output format (default:'
            ' %(default)s)', default='pdf')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
