from __future__ import division
from argparse import ArgumentParser, FileType
import matplotlib.pyplot as pp
import numpy as np

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
            print '%s,%g' % (args.params[i], np.corrcoef(X[:,i],y)[0,1])
    else:
        for i in xrange(d):
            print np.corrcoef(X[:,i],y)[0,1]
    # set ticks on x/y axis
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
        pp.savefig(args.output)
    pp.show()

if __name__ == '__main__':
    parser = ArgumentParser(description='plots scatters of data on parameter '\
            'values.')
    parser.add_argument('data', type=FileType('r'))
    parser.add_argument('-p', '--parameters', type=FileType('r'), help='set '\
            'titles to parameter names', dest='params_file')
    parser.add_argument('-o', '--output', type=FileType('w'), help='output file')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of '\
            'data values')
    parser.add_argument('-e','--error-bars', action='store_true', help='plot '\
            'error bars')
    parser.add_argument('-c','--columns', type=int, default=3, dest='cols')
    ns = parser.parse_args()
    main(ns)
