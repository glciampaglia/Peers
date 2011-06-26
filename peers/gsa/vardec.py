''' 
Computes main and total interaction effect indices using Winding Stairs (WS)
sampling and a Gaussian Process (GP) emulator
'''

import sys
import csv
from argparse import ArgumentParser, FileType
import numpy as np

from ..design.winding import wsinputs
from ..utils import SurrogateModel, gettxtdata

def indices(y):
    '''
    Computes main and total interaction effect indices from winding stairs (WS)
    matrix y
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

def main(args):
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(args.data.read(5000))
    args.data.seek(0)
    reader = csv.DictReader(args.data, dialect=dialect)
    paramnames = reader.fieldnames[:-args.responses]
    responsenames = reader.fieldnames[-args.responses:]
    args.params = reader.fieldnames
    args.data.seek(0)
    if args.with_errors:
        X, Y, _ = gettxtdata(args.data, args.responses,
                delimiter=dialect.delimiter, with_errors=True, skiprows=1)
    else:
        X, Y = gettxtdata(args.data, args.responses,
                delimiter=dialect.delimiter, with_errors=False, skiprows=1)
    N, M = X.shape
    intervals = zip(X.min(axis=0), X.max(axis=0))
    prng = np.random.RandomState(args.seed)
    Winput = wsinputs(args.size, M, intervals, prng)
    sm = SurrogateModel.fitGP(X,Y)
    YW = np.dstack([ sm(Winput[i]).T for i in xrange(len(Winput))]).swapaxes(1,2)
    outnames = [ 'variable', 'variance' ] + paramnames
    writer = csv.DictWriter(sys.stdout, outnames, dialect=dialect)
    mainrows = []
    interrows = []
    for resp, yw in zip(responsenames, YW):
        main, inter, totvar = indices(yw)
        row = { 'variable' : resp, 'variance' : totvar }
        row.update(zip(paramnames, main))
        mainrows.append(dict(row))
        row.update(zip(paramnames, inter))
        interrows.append(dict(row))
    print '; main effects'
    writer.writerow(dict(zip(writer.fieldnames, writer.fieldnames)))
    writer.writerows(mainrows)
    print '; interaction effects'
    writer.writerow(dict(zip(writer.fieldnames, writer.fieldnames)))
    writer.writerows(interrows)

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data', type=FileType('r'), help='data file')
    parser.add_argument('responses', type=int, help='number of responses')
    parser.add_argument('-s', '--size', help='number of samples (default:'
            ' %(default)d)', type=int, default=64)
    parser.add_argument('-e','--with-errors', action='store_true', help='if TRUE'\
            ', interprete last field as measurement standard errors')
    parser.add_argument('seed', type=int, nargs='?', help='Seed for the '
            'generator of pseudo-random numbers')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
