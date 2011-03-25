''' winding stairs sampling '''

import numpy as np
from argparse import ArgumentParser, FileType, Action

def winditer(rows, dims, prng=np.random):
    '''
    Returns an iterator over a winding stairs design sample, with U[0,1]
    marginal distributions. 

    Parameters
    ----------
    rows - number of rows of the winding stairs matrix
    dims - number of parameters
    '''
    r = prng.uniform(size=dims)
    yield tuple(r)
    for i in xrange(1,rows*dims):
        k = i % dims
        r[k] = prng.uniform()
        yield tuple(r)

def wsinputs(rows, dims, intervals=None, prng=np.random):
    '''
    Returns a rows x dims x dims array of inputs of a winding stairs design
    '''
    ws = np.array(list(winditer(rows, dims, prng)))
    ws = ws.reshape((rows, dims, dims))
    if intervals is not None:
        ws = ws * np.diff(intervals, axis=0) + intervals[0]
    return ws

def main(args):
    args.prng = np.random.RandomState(args.seed)
    if args.dims is None:
        args.dims = len(args.intervals)
    for i in xrange(args.dims - len(args.intervals)):
        args.intervals.append((0.,1.))
    args.intervals = np.array(args.intervals).T
    ws = wsinputs(args.rows, args.dims, args.intervals, args.prng)
    if args.binary_file is not None:
        np.save(args.binary_file, ws)
    else:
        for r in xrange(args.rows):
            for d in xrange(args.dims):
                print args.sep.join(map(str,ws[r,d]))

class Append(Action):
    def __call__(self, parser, ns, values, option_string=None):
        a, b = values
        if a > b:
            parser.error('illegal interval: %g %g' % (a, b))
        getattr(ns, self.dest).append((a, b))

if __name__ == '__main__':
    parser = ArgumentParser(description='winding stairs sampling.', epilog='by '
            'default, prints tuple of inputs to standard output')
    parser.add_argument('rows', help='number of rows', type=int)
    parser.add_argument('dims', help='number of variables', type=int, nargs='?')
    parser.add_argument('-s','--seed', help='random number generator\'s seed', 
            type=int)
    parser.add_argument('-i', '--interval', nargs=2, type=float, action=Append,
            dest='intervals', metavar='VALUE', default=[], help='set '
            'the n-th parameter to have values in interval (VALUE, VALUE). '
            'NOTE: You can pass this option more than once.')
    parser.add_argument('-d', '--delimiter', dest='sep', default=',', 
            metavar='CHAR', help='output fields are separated by %(metavar)s')
    parser.add_argument('-b', '--binary', dest='binary_file', help='produce '
            'a binary NPY file instead of writing to standard output',
            type=FileType('w'), metavar='FILE')
    ns = parser.parse_args()
    if len(ns.intervals) == 0 and ns.dims is None:
        parser.error('you must either specify the dimension or the intervals.')
    main(ns)