#!/usr/bin/python

'''
Produces main and interaction effect plots for global sensitivity analysis.
'''

from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp
from matplotlib.lines import lineMarkers
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from string import uppercase

from ..utils import sanetext, SurrogateModel, rect, fmt

lineMarkers = lineMarkers.items()
lineMarkers = filter(lambda k : k[1] != '_draw_nothing', lineMarkers)
lineMarkers = filter(lambda k : not isinstance(k[0], int), lineMarkers)
lineMarkers, _ = zip(*lineMarkers)

def maineffect(surrogate, bounds, num=10000):
    bounds = np.asarray(bounds)
    N = len(bounds)
    result = []
    for i in xrange(N):
        xm, xM = bounds[i]
        one = np.ones(num)
        effect_col = []
        s = np.diff(bounds, axis=1).T
        m = bounds[:, 0]
        Y = []
        X = np.linspace(xm, xM, 20, endpoint=True)
        for x in X:
            rvs = np.random.uniform(size=(num,N)) * s + m
            rvs[:,i] = one * x
            Y.append(surrogate(rvs).mean(axis=0))
        result.append((X, np.asarray(Y)))
    return map(np.asarray, zip(*result))

def twowayeffect(i, j, surrogate, bounds, num=10000):
    if i == j:
        raise ValueError('please provide two distinct variables')
    bounds = np.asarray(bounds)
    N = len(bounds)
    n = np.arange(N)
    isl = slice(bounds[i,0], bounds[i,1], 20j)
    jsl = slice(bounds[j,0], bounds[j,1], 20j)
    Xi, Xj = np.mgrid[isl,jsl]
    s = np.diff(bounds, axis=1).T
    m = bounds[:, 0]
    one = np.ones(num)
    Y = []
    for xi, xj in zip(Xi.ravel(), Xj.ravel()):
        rvs = np.random.uniform(size=(num,N)) * s + m
        rvs[:, i] = one * xi
        rvs[:, j] = one * xj
        y = surrogate(rvs).mean(axis=0)
        Y.append(y)
    Y = np.reshape(Y, Xi.shape)
    return Xi,Xj,Y

def plotmain(X, Y, names=None, output=None):
    '''
    Plots the main effect of all factors

    Parameters
    ----------
    X      - a list of arrays with factors value
    Y      - a list of arrays with main effect values
    output - file-like object
    '''
    pp.close('all')
    Mi, N, Mo = Y.shape
    rows, cols =rect(Mo)
    fig = pp.figure()
    for k in xrange(Mo):
        ax = fig.add_subplot(rows, cols, k+1)
        for i, (x, y) in enumerate(zip(X, Y[...,k])):
            x = (x - x.min()) / (x.max() - x.min()) 
            l, = pp.plot(x, y, hold=1, color='k', alpha=.75, 
                    marker=lineMarkers[i])
            if names is not None:
                l.set_label(sanetext(names[i]))
        pp.xlabel('parameter scaled value')
        pp.ylabel('main effect')
    ym, yM = pp.ylim()
    yl = yM - ym
    if Mo > 1:
        pp.axis('tight')
        pp.ylim(ym, yM + .15 * yl)
        ax.text(0.05, 0.95, uppercase[k], transform=ax.transAxes, 
                fontsize='x-large', fontweight='bold', va='top')
    if names is not None:
        pp.axis('tight')
        pp.ylim(ym, yM + .25 * yl)
        pp.legend(ncol=2, loc='upper center', markerscale=.8,
                prop=FontProperties(size=7), columnspacing=0.1)
# XXX provide parameter dict
#    fig.subplots_adjust()
    pp.draw()
    if output is not None:
        pp.savefig(output, format=fmt(output.name))
    pp.show()

def plottwoway(Xi, Xj, Y, labels=None, incolor=False, output=None):
    '''
    Plots the 2-way interaction effect between given factors.

    Parameters
    ----------
    Xi, Xj, Y - 2D data arrays
    labels    - specify axes labels
    incolor   - produce a color plot with a color bar
    '''
    pp.close('all')
    fig = pp.figure()
    ax = Axes3D(fig)
    if incolor:
        surf = ax.plot_surface(Xi, Xj, Y, rstride=1, cstride=1, cmap=pp.cm.jet,
                antialiased=True, linewidth=1)
        fig.colorbar(surf, shrink=0.5, aspect=5)
    else:
        surf = ax.plot_surface(Xi, Xj, Y, rstride=1, cstride=1, color='w',
                antialiased=True, linewidth=1)
    if labels is not None:
        xlabel, ylabel, zlabel = map(sanetext, labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
    else:
        ax.set_xlabel('parameter A value')
        ax.set_ylabel('parameter B value')
        ax.set_zlabel('interaction effect')
    for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
        for l in axis.get_ticklabels():
            l.set_fontsize(5)
    pp.draw()
    if output is not None:
        pp.savefig(output, format=fmt(output.name))
    pp.show()
    pp.close()

def main(args):
    if args.parameters is not None:
        parameters = args.parameters.readline().strip().split(args.delimiter)
    else:
        parameters = None
    data = np.loadtxt(args.data, delimiter=args.delimiter)
    if args.with_errors:
        M = args.responses * 2
        X = data[:,:-M]
        Y = data[:,-M::2]
        Ye = data[:, -M+1::2]
    else:
        X = data[:,:-args.responses]
        Y = data[:,-args.responses:]
        Ye = None
    sm = SurrogateModel.fitGP(X, Y)
    bounds = zip(X.min(axis=0), X.max(axis=0))
    if args.main:
        Xm, Ym = maineffect(sm, bounds, args.num)
        plotmain(Xm, Ym, parameters, args.output)
        return Xm, Ym
    elif args.interaction is not None:
        i, j = args.interaction
        Xi, Xj, Y = twowayeffect(i, j, sm, bounds, args.num)
        plottwoway(Xi, Xj, Y, args.labels, args.color, args.output)
        return Xi, Xj, Y

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data', type=FileType('r'), help='simulation data')
    parser.add_argument('responses', type=int, help='number of response '
            'variables (default: %(default)d)', default=1)
    parser.add_argument('-d', '--delimiter', default=',', metavar='CHAR',
            help='data fields are delimited by %(metavar)s (default: '
            '\'%(default)s\')')
    parser.add_argument('-e', '--with-errors', help='data contains measurement'
            ' errors', action='store_true')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-m', '--main', action='store_true', help='plot main '
            'effects')
    group.add_argument('-i', '--interaction', type=int, nargs=2, metavar='VAR',
            help='plot interaction effects for given pair of %(metavar)ss', )
    parser.add_argument('-n', '--num', type=int, default=1000, help='sample size '
            'for monte carlo (default: %(default)d)')
    parser.add_argument('-p', '--parameters', help='parameter names',
            type=FileType('r'), metavar='FILE')
    parser.add_argument('-l', '--labels', help='Axes label for 3D interaction '
            'effect plot', nargs=3, metavar='TEXT')
    parser.add_argument('-c', '--color', help='Produce a colored 3D interaction'
            ' effect plot', action='store_true')
    parser.add_argument('-o', '--output', type=FileType('w'), help='graphics '
            'output file', metavar='FILE')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    res = main(ns)
