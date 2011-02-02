'''
Computes sensitivity indices as standardized linear regression coefficients.
'''

from argparse import ArgumentParser, FileType
import numpy as np
import scikits.statsmodels as sm
import matplotlib.pyplot as pp

def main(args):
    data = np.loadtxt(args.input, delimiter=args.sep)
    if args.params_file is not None:
        args.params = args.params_file.readline().strip().split(',')
    else:
        args.params = None
    if args.with_error:
        x = data[:,:-2]
        y = data[:,-2]
        ye = data[:,-1]
    else:
        x = data[:,:-1]
        y = data[:,-1]
        ye = None
    x = (x - x.mean(axis=0))/x.std(axis=0)
    y = (y - y.mean())/y.std()
    res = sm.GLS(y, x).fit()
    print res.summary(xname=args.params)

if __name__ == '__main__':
    parser = ArgumentParser(description='Computes sensitivity indices using '\
            'multiple linear regression')
    parser.add_argument('input', type=FileType('r'), help='data file.', 
            metavar='FILE')
    parser.add_argument('-p', '--parameters', type=FileType('r'), help=
            'read comma-separated list of parameter names from %(metavar)s',
            dest='params_file', metavar='FILE')
    parser.add_argument('-d', '--delimiter', dest='sep', default=',', 
            metavar='CHAR', help='input data fields are separated by '
            '%(metavar)s (default: "%(default)s)"')
    parser.add_argument('-e','--with-error', action='store_true', help='if TRUE'\
            ', interprete last field as measurement standard errors')
    ns = parser.parse_args()
    main(ns)
    
