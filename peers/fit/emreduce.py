#!/usr/bin/env python

'''Reduction script for Gaussian Mixture Model (GMM) parameter estimation.'''

import os
import sys
from itertools import groupby
from argparse import ArgumentParser, FileType
import numpy as np
from scikits.learn.mixture import GMM

def main(args):
    ''' reads index, groups by parameters '''
    if not os.path.exists(args.directory) or not os.path.isdir(args.directory):
        raise ValueError('not a directory: %s' % args.directory)
    liter = ( tuple(l.strip().split(args.sep)) for l in iter(args.index.readline,'') )
    for k, subiter in groupby(liter, lambda k : k[:-1]):
        beta = []
        for fn in ( os.path.join(args.directory, s[-1]) for s in subiter ):
            d = np.load(fn)
            if len(d):
                gmm = GMM(args.components)
                gmm.fit(d)
                mu, si, we = map(np.ravel, 
                        [gmm.means, np.asarray(gmm.covars), gmm.weights])
                idx = mu.argsort()
                beta.append(np.hstack([mu[idx], si[idx], we[idx]]))
        if len(beta) == 0:
            if args.verbose:
                print >> sys.stderr, 'NO DATA: %s' % ','.join(map(str,k))
            continue
        beta = np.asarray(beta).mean(axis=0)
        out = k + tuple(beta)
        fmt = ','.join(['%s'] * len(out))
        print fmt % out

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('index', type=FileType('r'), help='Data files index.')
    parser.add_argument('components', type=int, help='Number of GMM components.')
    parser.add_argument('-C', '--directory', default='.', help='Interpret '
            'file paths in index as relative to %(metavar)s.', metavar='DIR')
    parser.add_argument('-d', '--delimiter', default=',', help='Data files index'
            ' has fields separated by %(metavar)s.', dest='sep', metavar='CHAR')
    parser.add_argument('-v', '--verbose', action='store_true', help='Be '
            'verbose. Warn when encountering empty data files.')
    ns = parser.parse_args()
    main(ns)
