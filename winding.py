''' winding stairs sampling '''

import numpy as np
from argparse import ArgumentParser, FileType, Action

def windindex(n,d):
    '''
    returns an iterator over a sequence of indices of a winding stairs matrix
    for a sample of n observations of d variables.

    Parameters
    ----------
    n, d - shape of the matrix
    '''
    idx = np.ndindex(n,d)
    for i in xrange(n):
        item = [ idx.next() for j in xrange(d) ]
        yield zip(*item)
        for j in xrange(1,d):
            new_item = list(item)
            k = new_item[j]
            if k[0]+1 < n:
                new_item[j] = k[0]+1, k[1]
                item = new_item
                yield zip(*new_item)
            else:
                raise StopIteration

def winding(arr):
    '''
    Takes a sample of d model input variables of size n and returns an iterator
    over the winding stairs matrix of the sample

    Parameters
    ----------
    arr - input sample
    '''
    arr = np.atleast_2d(arr)
    if arr.ndim > 2:
        raise ValueError('arr.ndim > 2')
    for ii, jj in windindex(*arr.shape):
        yield arr[ii,jj]

def main(args):
    if args.dim is None:
        args.dim = len(args.intervals)
    for i in xrange(args.dim - len(args.intervals)):
        args.intervals.append((0.,1.))
    sample = []
    for l, h in args.intervals:
        sample.append(np.random.uniform(l, h, size=args.size))
    sample = np.asarray(sample).T
    for p in winding(sample):
        print args.sep.join(map(str,p))

class Append(Action):
    def __call__(self, parser, ns, values, option_string=None):
        a, b = values
        if a > b:
            parser.error('illegal interval: %g %g' % (a, b))
        getattr(ns, self.dest).append((a, b))

if __name__ == '__main__':
    parser = ArgumentParser(description='winding stairs sampling.')
    parser.add_argument('size', help='sample size', type=int)
    parser.add_argument('dim', help='sample dimension', type=int, nargs='?')
    parser.add_argument('-i', '--interval', nargs=2, type=float, action=Append,
            dest='intervals', metavar='VALUE', default=[], help='set '
            'bounds of i-th parameter to (VALUE, VALUE). NOTE: You can pass '
            'this option more than once.')
    parser.add_argument('-d', '--delimiter', dest='sep', default=',', 
            metavar='CHAR', help='output fields are separated by %(metavar)s')
    ns = parser.parse_args()
    if len(ns.intervals) == 0 and ns.dim is None:
        parser.error('you must either specify the dimension or the intervals.')
    main(ns)
