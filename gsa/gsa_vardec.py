from argparse import ArgumentParser, FileType
import numpy as np

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

def main(args):
    data = np.loadtxt(args.data, delimiter=args.sep)
    if args.params_file is not None:
        args.params = args.params_file.readline().strip().split(',')
    if args.with_error:
        x,y = makews(data[:,:-1])
    else:
        x,y = makews(data)
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
    print 'Total variance : %g' % s
    if args.params_file is not None:
        for p,m,t in zip(args.params,main,inter):
            print '%s : %g\t%g' % (p, 1 - m / s, t / s)
    else:
        for m, t in zip(main,inter):
            print '%g\t%g' % (1 - m / s, t / s)
        

if __name__ == '__main__':
    parser = ArgumentParser(description='Computes main and interaction effect '
            'indices')
    parser.add_argument('data', type=FileType('r'), metavar='FILE', help='data'\
            ' file')
    parser.add_argument('-p', '--parameters', type=FileType('r'), help='set '\
            'titles to parameters taken from %(metavar)s', dest='params_file', 
            metavar='FILE')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of '\
            'data values', metavar='CHAR', dest='sep')
    parser.add_argument('-e','--with-error', action='store_true', help='if TRUE'\
            ', interprete last field as measurement standard errors')
    ns = parser.parse_args()
    main(ns)
