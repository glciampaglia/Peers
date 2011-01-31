from __future__ import division
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp

def pcc(x,y,z):
    '''
    x,y - (N,) arrays of variables
    z   - (M,N) array of M confounding variables

    Computes the partial correlation coefficient between x and y, given z
    '''
    zorth = np.linalg.qr(z.T)[0].T
    x = x - np.dot(np.dot(zorth, x), zorth)
    y = y - np.dot(np.dot(zorth, y), zorth)
    return np.dot(x,y)/np.sqrt(np.dot(x,x) * np.dot(y,y))

def main(args):
    data = np.loadtxt(args.data, delimiter=args.delimiter)
    fig = pp.figure()
    # set space between plots
    fig.subplotpars.update(wspace=.2,hspace=.5)
    if args.error_bars:
        X = data[:,:-2]
        y = data[:,-2]
        ye = data[:,-1]
        n,d = X.shape
        if d < args.cols:
            args.cols = d
        args.rows = np.ceil(d / args.cols)
        for i,x in enumerate(X.T):
            ax = pp.subplot(args.rows,args.cols,i+1)
            ax.errorbar(x,y,ye/2.0, marker='o', ls='',c='w')
    else:
        X = data[:,:-1]
        y = data[:,-1]
        n,d = X.shape
        if d < args.cols:
            args.cols = d
        args.rows = np.ceil(d / args.cols)
        for i,x in enumerate(X.T):
            ax = pp.subplot(args.rows,args.cols,i+1)
            ax.scatter(x,y,c='w')
    # set titles
    if args.params_file is not None:
        args.params = args.params_file.readline().strip().split(',')
        for i in xrange(d):
            ax = fig.axes[i]
            ax.set_title(args.params[i].replace('_',' ').capitalize())
            idx = range(d)
            del idx[i]
            print '%s,%g' % (args.params[i], pcc(X[:,i], y, X[:,idx].T))
    else:
        for i in xrange(d):
            idx = range(d)
            del idx[i]
            print pcc(X[:,i],y,z[:,idx].T)
    # set ticks on x/y axis, y label only on first plot
    for i in xrange(d):
        ax = fig.axes[i]
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(np.linspace(xmin,xmax,3))
        ymin, ymax = ax.get_ylim()
        ax.set_yticks(np.linspace(ymin,ymax,5))
        if i > 0:
            ax.set_yticklabels([])
        if i == 0:
            ax.set_ylabel(r'$<\tau>$ (days)',fontsize=14)
    if args.output is not None:
        for out in args.output:
            pp.savefig(out)
    pp.show()

def make_parser():
    parser = ArgumentParser(description='plots scatters of data on parameter '\
            'values.')
    parser.add_argument('data', type=FileType('r'), metavar='FILE', help='data'\
            ' file')
    parser.add_argument('-p', '--parameters', type=FileType('r'), help='set '\
            'titles to parameters taken from %(metavar)s', dest='params_file', 
            metavar='FILE')
    parser.add_argument('-o', '--output', type=FileType('w'), help='save '\
            'graphics in %(metavar)s', nargs='+', metavar='FILE')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of '\
            'data values', metavar='CHAR')
    parser.add_argument('-e','--error-bars', action='store_true', help='plot '\
            'error bars (standard errors in last column)')
    parser.add_argument('-c','--columns', type=int, default=3, dest='cols',
            help='arrange plots in %(metavar)s columns', metavar='NUM')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
