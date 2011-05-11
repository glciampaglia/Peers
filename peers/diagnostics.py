''' plots the number of users and pages '''

from argparse import ArgumentParser, FileType

import numpy as np
from scipy.interpolate import splrep, splev
from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter
import matplotlib.pyplot as pp
from collections import deque
import sys

from gsa.utils import fmt

def counts(x, span, tmin=0.):
    '''
    Computes counts with given frequency

    Parameters
    ----------
    x    - a 2D array sorted along axis 0 (time)
    span - time frequency span (in days)
    tmin - (optional) initial time, instead of tmin = 0.
    '''
    if span <= 0:
        raise ValueError('positive span expected : %g' % span)
    N, M = x.shape
    tmax = np.ceil(x[-1, 0])
    Nc = int((tmax - tmin) / span)
    res = np.empty((Nc, M))
    nums, T = np.histogram(x[:, 0], bins=Nc, range=(tmin,tmax))
    c = 0
    for i, n in enumerate(nums):
        res[i, 1:] = x[c:(c + n), 1:].mean(axis=0)
        res[i, 0] = T[i]
        c += n
    return res

def counts_main(args):
    coll = {}
    for i, f in enumerate(args.inputs):
        print 'processing %s ...' % f.name,
        sys.stdout.flush()
        x = np.loadtxt(f)
        coll[f.name] = counts(x, args.frequency)
        print 'done'
        sys.stdout.flush()
    print 'saving to %s ...' % args.output.name,
    sys.stdout.flush()
    np.savez(args.output, **coll)
    print 'done.'

def plot_main(args):
    if args.rescale is not None and args.rescale <= 0:
        raise ValueError('expected positive value for rescaling : %d' %
                args.rescale)
    coll = []
    npz = np.load(args.input)
    for arr in ( npz[n] for n in sorted(npz.files) ):
        x, U, P = arr.T
        if args.plot_users:
            y = U
        else:
            y = P
        if args.rescale is not None:
            y = y / y[-args.rescale:].mean()
        coll.append(np.c_[x,y])
    fig = pp.figure()
    ax = pp.axes([0.1, 0.15, 0.85, 0.80])
    lc = LineCollection(coll, 
            colors=colorConverter.to_rgba_array('k' * len(coll), args.alpha))
    ax.add_collection(lc)
    if args.mean:
        coll = np.asarray(coll)
        m = coll[:,::20,1].mean(axis=0)
        spl = splrep(x[::20], m)
        tt = np.linspace(x[0],x[-1], 1000)
        mm = splev(tt, spl)
        ax.plot(tt, mm, 'r-', lw=3)
    pp.axis('tight')
    if args.annotate is not None:
        ym, yM = pp.ylim()
        ys = yM - ym
        xt = args.annotate * .45
        yt = yM + .1 * ys
        ax.text(xt, yt, 'transient', fontsize='small')
        ax.axvspan(0, args.annotate, color='b', alpha=args.alpha/2)
        pp.ylim(ym, yM + .2 * ys)
    pp.xlabel('time (days)')
    if args.rescale is not None:
        pp.ylabel(r'scaled number of users $N_u / \overline{N_u}$')
    else:
        pp.ylabel(r'number of users $N_u$')
    pp.draw()
    if args.output is not None:
        pp.savefig(args.output, format=fmt(args.output.name))
    pp.show()

def main(args):
    return args.action(args)

def make_parser():
    parser = ArgumentParser()
    subs = parser.add_subparsers(help='action to perform')
    #
    # parser for producing counts data
    parser_a = subs.add_parser('counts', help='compute counts data')
    parser_a.add_argument('-f', '--frequency', type=float, default=1.0,
            help='frequency in days (default: %(default)g)', metavar='VALUE')
    parser_a.add_argument('inputs', metavar='FILE', help='input file(s)', 
            nargs='+', type=FileType('r'))
    parser_a.add_argument('-o','--output', help='output file', required=True,
            type=FileType('w'))
    parser_a.set_defaults(action=counts_main)
    #
    # parser for plotting counts data
    parser_b = subs.add_parser('plot', help='plot counts data')
    group = parser_b.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--pages', action='store_false', default=False,
            help='plot only number of pages', dest='plot_users')
    group.add_argument('-u', '--users', action='store_true', default=False,
            help='plot only number of users', dest='plot_users')
    parser_b.add_argument('input', type=FileType('r'), help='input archive '
            'file', metavar='FILE')
    parser_b.add_argument('-o', '--output', metavar='FILE', type=FileType('w'),
            help='save graphics to %(metavar)s')
    parser_b.add_argument('-a', '--alpha', type=float, help='alpha value',
            default=0.5)
    parser_b.add_argument('-m', '--mean', action='store_true', help='add mean'
            ' of data to plot')
    parser_b.add_argument('-r', '--rescale', type=int, help='rescale each curve'
            ' independently to mean over last %(metavar)s observations',
            metavar='NUM')
    parser_b.add_argument('-n', '--annotate', type=int, help='add shaded '
            'regions at t = %(metavar)s', metavar='NUM')
    parser_b.set_defaults(action=plot_main)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
