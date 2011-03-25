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

from .utils import SurrogateModel, rect

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
        raise ValueError('please two distinct variables')
    bounds = np.asarray(bounds)
    N = len(bounds)
    n = np.arange(N)
    isl = slice(bounds[i,0], bounds[i,1], 20j)
    jsl = slice(bounds[i,0], bounds[i,1], 20j)
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

def plotmain(X, Y, names=None):
    pp.close('all')
    Mi, N, Mo = Y.shape
    rows, cols =rect(Mo)
    fig = pp.figure(figsize=(4 * cols, 4 * rows))
    for k in xrange(Mo):
        ax = fig.add_subplot(rows, cols, k+1)
        for i, (x, y) in enumerate(zip(X, Y[...,k])):
            x = (x - x.min()) / (x.max() - x.min()) 
            l, = pp.plot(x, y, hold=1, color='k', alpha=.75, 
                    marker=lineMarkers[i])
            if names is not None:
                l.set_label(names[i])
        pp.xlabel('parameter scaled value', fontsize=14)
        pp.ylabel('main effect', fontsize=14)
    ym, yM = pp.ylim()
    yl = yM - ym
    if Mo > 1:
        pp.axis('tight')
        pp.ylim(ym, yM + .15 * yl)
        ax.text(0.05, 0.95, uppercase[k], transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top')
    if names is not None:
        pp.axis('tight')
        pp.ylim(ym, yM + .25 * yl)
        pp.legend(ncol=len(names)/3, loc='upper center', markerscale=.5,
                prop=FontProperties(size='x-small'))
    fig.subplots_adjust(wspace=.25, left=.15, right=.95, bottom=.15)
    pp.draw()
    pp.show()

def plottwoway(Xi, Xj, Y, xlabel=None, ylabel=None):
    pp.close('all')
    fig = pp.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(Xi, Xj, Y, rstride=1, cstride=1, cmap=pp.cm.jet,
            linewidth=2, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if xlabel is not None:
        pp.xlabel(xlabel, fontsize=14)
    else:
        pp.xlabel('parameter value', fontsize=14)
    if xlabel is not None:
        pp.ylabel(ylabel, fontsize=14)
    else:
        pp.ylabel('main effect', fontsize=14)
    pp.draw()
    pp.show()

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
    # TODO get these from file instead?
    bounds = zip(X.min(axis=0), X.max(axis=0))
    if args.main:
        Xm, Ym = maineffect(sm, bounds, args.num)
        plotmain(Xm, Ym, parameters)
        return Xm, Ym
    elif args.interaction is not None:
        i, j = args.interaction
        Xi, Xj, Y = twowayeffect(i, j, sm, bounds, args.num)
        plottwoway(Xi, Xj, Y, args.xlabel, args.ylabel)
        return Xi, Xj, Y

if __name__ == '__main__':
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
    parser.add_argument('-x', '--xlabel', help='X axis label for interaction '
            'effect plot')
    parser.add_argument('-y', '--ylabel', help='Y ayis label for interaction '
            'effect plot')
    ns = parser.parse_args()
    res = main(ns)

