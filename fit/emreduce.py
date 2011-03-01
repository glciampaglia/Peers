#!/usr/bin/env python

'''
GMM reduce script
'''

import os
import sys
from itertools import groupby
from argparse import ArgumentParser, FileType
import numpy as np
from scikits.learn import mixture as m

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
                gmm = m.GMM(args.components)
                gmm.fit(d)
                beta.append(np.hstack(map(np.ravel,
                    [ gmm.means, np.asarray(gmm.covars), gmm.weights ])))
        if len(beta) == 0:
            if args.verbose:
                print >> sys.stderr, 'NO DATA: %s' % ','.join(map(str,k))
            continue
        beta = np.asarray(beta).mean(axis=0)
        out = k + tuple(beta)
        fmt = ','.join(['%s'] * len(out))
        print fmt % out

if __name__ == '__main__':
    parser = ArgumentParser(description='Computes Gaussian Mixture Model (GMM) '
            'parameters')
    parser.add_argument('index', type=FileType('r'), help='index file')
    parser.add_argument('components', type=int, help='number of GMM components')
    parser.add_argument('-C', '--directory', default='.', help='data directory')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of '\
            'index lines', dest='sep')
#     parser.add_argument('-s', '--standard-error', action='store_true', 
#             help='output standard error values')
    parser.add_argument('-v', '--verbose', action='store_true', help='be '
            'verbose about empty files')
    ns = parser.parse_args()
    main(ns)
