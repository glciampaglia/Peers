#!/usr/bin/env python
# coding=utf-8
# file: lhd.py
# vim:ts=8:sw=4:sts=4

''' Latin Hypercube Designs. © 2010 Giovanni Luca Ciampaglia '''

# TODO: implement a simulated annealing procedure for finding a maximin design
# (see Morris, M. and Mitchell, T., 1995. Exploratory design for computer
# experiments. J. Statist. Plann. Inference 43, pp. 381–402.) 

import sys
from argparse import ArgumentParser
import numpy as np
from scipy.spatial.distance import pdist

from ..utils import AppendTuple

def _map_to_range(lhd, gr):
    lhd_idx = map(tuple, lhd)
    res = []
    for idx in lhd_idx:
        res.append(tuple([ gr[i][k] for i,k in enumerate(idx) ]))
    return np.asarray(res) + np.diff(gr[:,:2],axis=1).T/2

def lhd(m, n, num=None, ranges=None, prng=np.random, maximin=False):
    """
    latin hypercube design in m dimensions.

    Generate (indices of) centers of a latin hypercube design.

    Parameters
    ----------
    m   - non neg. scalar int
          number of dimensions
    n   - non neg. scalar int
          number of points
    num - non neg. scalar or None
          number of LHDs to generate
    ranges - list of m (a,b) tuples
          extrema of the intervals to map the centers into
    prng - instance of numpy.random.RandomState (default = numpy.random)
    maximin - boolean
        if True, returns the design that attains the maximum of mdist

    Returns
    -------
    lhd - (m,n) array
        LHD design
    mdist - float scalar
        minimum pairwise distance over the n points

    Notes
    -----
    By default, the function returns a list of indices into an (m x n) grid.
    If a list of interval bounds is passed, then their midpoints are returned
    instead. Multiple designs, together with the corresponding values of the
    pairwise minimum distance are returned if num is not None. The list may
    contain duplicate. 

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> x, y = np.mgrid[0:1:5j, 0:1:5j]
    >>> d, design = lhd(2,5, prng=prng)
    >>> print d
    1.41421356237
    >>> idx = map(tuple, design)    # list of indices for each grid
    >>> x[idx[0]], y[idx[0]]        # coordinates of the first center
    (0.5, 0.0)
    """
    if ranges is not None and len(ranges) != m:
        raise ValueError('expecting %d ranges' % m)
    if ranges is not None:
       gr = np.asarray([ np.linspace(a,b,n,endpoint=False) for (a,b) in
           ranges])
    else:
        gr = None
    if num is None:
        lhd = np.asarray([ prng.permutation(n) for i in xrange(m) ]).T
        if gr is not None:
            lhd = _map_to_range(lhd, gr)
        return pdist(lhd).min(), lhd
    else:
        lhd_iter = ( np.asarray([ prng.permutation(n) for i in xrange(m) ]).T
                for j in xrange(num) )
        if gr is not None:
            lhd_iter = ( _map_to_range(d, gr) for d in lhd_iter )
        lhd_iter = ( (pdist(d).min(), d) for d in lhd_iter )
        if maximin:
            max_d, max_design = -1, None
            for d, design in lhd_iter:
                if d > max_d:
                    max_d = d
                    max_design = design
            return max_d, max_design
        else:
            return list(lhd_iter)

def make_parser():
    parser = ArgumentParser(description='Latin hypercube sampling')
    parser.add_argument(
            'num',
            metavar='NUM',
            help='generate a sample with %(metavar)s points',
            type=int)
    parser.add_argument(
            'dimension',
            metavar='DIMS',
            help='number of dimensions of the sample',
            type=int)
    parser.add_argument(
            '-m',
            '--maximin',
            dest='maximin_num',
            metavar='NUM',
            help='maximize minimum distance over %(metavar)s designs',
            type=int)
    parser.add_argument(
            '-i',
            '--intervals',
            nargs=2,
            type=float,
            action=AppendTuple,
            metavar='VALUE',
            help='specify interval for i-th dimension. NOTE: can be passed '
            'multiple times',)
    parser.add_argument(
            '-s',
            '--seed',
            type=int,
            help='PRNG seed')
    parser.add_argument(
            '-S',
            '--separator',
            default=',',
            help='output separator (default: \'%(default)s\')')
    parser.add_argument(
            '-D',
            '--debug',
            action='store_true',
            help='raise Python exceptions to the console')
    parser.add_argument(
            '-r',
            '--repeat',
            type=int,
            default=1,
            metavar='NUM',
            help='repeat each input point %(metavar)s times')
    return parser

def repeat(n, pointsiter):
    for point in pointsiter:
        for i in xrange(n):
            yield point

def main(args):
    prng = np.random.RandomState(args.seed)
    args.maximin = (args.maximin_num is not None)
    dist, design = lhd(args.dimension, args.num, num=args.maximin_num, 
            ranges=args.intervals, maximin=args.maximin, prng=prng)
    for p in repeat(args.repeat, design):
        print args.separator.join(map(str,p))

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        main(ns)
    except:
        ty, val, tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            parser.error('%s : %s' % (ty.__name__, val))
