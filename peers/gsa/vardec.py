''' 
Computes main and total interaction effect indices using Winding Stairs (WS)
sampling and a Gaussian Process (GP) emulator
'''

from argparse import ArgumentParser, FileType
import numpy as np

from ..design.winding import wsinputs
from .utils import SurrogateModel, gettxtdata

# TODO: remove
def makews(data):
    '''
    Produces a "Winding Stairs Matrix"

    Parameters
    ----------
    data - N points sample
    '''
    data = np.atleast_2d(data)
    if data.ndim > 2:
        raise ValueError('input is not a 2-d array')
    N, k = data.shape
    d = k - 1
    r = np.floor(N / d) # rows of the W.S. matrix
    x = data[:,:-1]
    y = data[:,-1]
    return x.reshape((r,d,d)), y.reshape((r, d))

def indices(y):
    '''
    Computes main and total interaction effect indices out of matrix y
    '''
    n, d = y.shape
    main = []
    inter = []
    for i in xrange(d):
        # The main effect is estimated as the total variance minus half of the
        # average squared difference between elements that are d-1 away on the
        # right. Moving d-1 elements on the right in the design may cause a
        # "wraparound" to the next row. This means that for the first parameter
        # the main effect differences are taken between the first and the last
        # column, while for the second parameter (and similarly for all the
        # others) the differences are taken between the second and the first
        # column and, because of the wraparound, with elements of the first
        # column shifted by one row.
        ii = (i + d - 1) % d
        if i == 0:
            m = np.mean((y[:, i] - y[:, ii]) ** 2) / 2
        else:
            m = np.mean((y[:-1, i] - y[1:, ii]) ** 2) / 2
        main.append(m)
        # The interaction effect is half the average squared difference over
        # elements for which only one parameter value changes. This means that
        # along the sampling, the elements are adjacent in the sampling, i.e. 
        # one step on the right and optionally wrapping to the next row. For the
        # first parameter, the first change of value happens at the end of the
        # first row, so the differences are computed between elements of the
        # last and first column, with elements from the first column shifted
        # down by one row.
        # For all other parameters, the first change of value happens within
        # the first row, so in this case the differences are taken over two
        # adjacent and columns and rows are aligned.  
        if i == 0:
            t = np.mean((y[:-1, -1] - y[1:, 0]) ** 2) / 2
        else:
            t = np.mean((y[:, i - 1] - y[:, i]) ** 2) / 2
        inter.append(t)
    # The coefficients are just the main/total effects normalized by the total
    # variance. Within each column, the samples are indipendent, so the total
    # variance is obtained by pooling the variances estimated along each column
    s = y.var(axis=0, ddof=1).mean()
    main = 1.0 - np.asarray(main) / s
    inter = np.asarray(inter) / s
    return main, inter, s

# TODO <Fri Mar 25 23:45:37 CET 2011>: should get names of response variables
# from command line (or file)
def main(args):
    if args.params_file is not None:
        args.params = args.params_file.readline().strip().split(args.sep)
    if args.with_errors:
        X, Y, _ = gettxtdata(args.data, args.responses, delimiter=args.sep,
                with_errors=True)
    else:
        X, Y = gettxtdata(args.data, args.responses, delimiter=args.sep,
                with_errors=False)
    N, M = X.shape
    intervals = zip(X.min(axis=0), X.max(axis=0))
    prng = np.random.RandomState(args.seed)
    W = wsinputs(args.rows, M, intervals, prng)
    sm = SurrogateModel.fitGP(X,Y)
    YW = np.dstack([ sm(W[i]).T for i in xrange(len(W))]).swapaxes(1,2)
    for i,yw in enumerate(YW):
        print '-------------'
        print 'Parameter %d' % i
        print '-------------'
        main, inter, s = indices(yw)
        print 'Total variance : %g' % s
        if args.params_file is not None:
            for p,m,t in zip(args.params, main, inter):
                print '%s\t%g\t%g' % (p, m, t)
        else:
            for m, t in zip(main, inter):
                print '%g\t%g' % (m, t)
        print

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data', type=FileType('r'), help='data file')
    parser.add_argument('responses', type=int, help='number of response '
            'variables (default: %(default)d)', default=1)
    parser.add_argument('-r', '--rows', help='number of WS rows (default:'
            ' %(default)d)', type=int, default=64)
    parser.add_argument('-p', '--parameters', type=FileType('r'), help='set '\
            'titles to parameters taken from %(metavar)s', dest='params_file', 
            metavar='FILE')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of '\
            'data values', metavar='CHAR', dest='sep')
    parser.add_argument('-e','--with-errors', action='store_true', help='if TRUE'\
            ', interprete last field as measurement standard errors')
    parser.add_argument('seed', type=int, nargs='?', help='Seed for the '
            'generator of pseudo-random numbers')
    ns = parser.parse_args()
    main(ns)
