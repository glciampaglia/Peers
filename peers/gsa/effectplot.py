#!/usr/bin/python

'''
Main and interaction effect plots 
'''

from argparse import ArgumentParser, FileType
import numpy as np
from scikits.learn.gaussian_process import GaussianProcess
import matplotlib.pyplot as pp
from matplotlib.lines import lineMarkers
from matplotlib.font_manager import FontProperties
from matplotlib.cm import jet
from mpl_toolkits.mplot3d import Axes3D
from string import uppercase

lineMarkers = lineMarkers.items()
lineMarkers = filter(lambda k : k[1] != '_draw_nothing', lineMarkers)
lineMarkers = filter(lambda k : not isinstance(k[0], int), lineMarkers)
lineMarkers = dict(lineMarkers).keys()

class SurrogateModel(object):
    '''
    A class that evaluates a gaussian model independently for each response
    variable
    '''
    def __init__(self, models):
        self._models = models
    @property
    def models(self):
        return list(self._models)
    def __call__(self, X, **kwargs):
        '''
        Calls GaussianProcess.predict on each of the GP models of the instance.
        Additional keyword arguments are passed to it. 
        '''
        Y = [ m.predict(X, **kwargs) for m in self._models ]
        return np.row_stack(Y).T
    @classmethod
    def fitGP(cls, X, Y, **kwargs):
        '''
        Fits a gaussian process of X to each variable in Y and returns a
        SurrogateModel instance
        '''
        models = []
        for y in Y.T:
            gp = GaussianProcess(**kwargs)
            gp.fit(X, y)
            models.append(gp)
        return cls(models)
    def __repr__(self):
        return repr(self.models)

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

# XXX must take indices to plot
def interactioneffect(surrogate, bounds, num=10000):
    bounds = np.asarray(bounds)
    N = len(bounds)
    result = []
    for i,j in np.ndindex(N,N):
        if i >= j:
            continue
        n = np.arange(N)
        isl = slice(bounds[i,0], bounds[i,1], num*1j)
        jsl = slice(bounds[i,0], bounds[i,1], num*1j)
        Xi,Xj = np.mgrid[isl,jsl]
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
        result.append((Xi,Xj,Y))
    return result

def rect(x):
    ''' Finds arg min_{m, n <= x and m * n >= x}{ m - n} '''
    x = int(x)
    if x == 1:
        return (1,1)
    if x == 2:
        return (1,2)
    allrows, = np.where(np.mod(x, np.arange(1,x+1)))
    allcols = np.asarray(np.ceil(float(x) / allrows), dtype=int)
    i = np.argmin(np.abs(allrows - allcols))
    return allrows[i], allcols[i]

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

# XXX will do only ONE plot
def plotint(data, names=None):
    pp.close('all')
    for Xi, Xj, Y in data:
        fig = pp.figure()
        ax = Axes3D(fig)
        l = ax.plot_surface(Xi, Xj, Y, rstride=4, cstride=4, color='b')
        pp.xlabel('parameter scaled value', fontsize=14)
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
    bounds = zip(X.min(axis=0), X.max(axis=0))
    if args.main:
        Xm, Ym = maineffect(sm, bounds, args.num)
        plotmain(Xm, Ym, parameters)
        return Xm, Ym
    if args.interaction:
        raise NotImplementedError('yet to be finished')
        res = interactioneffect(sm, bounds, args.num)
        plotint(res)
        return res

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
    group.add_argument('-i', '--interaction', action='store_true', help='plot '
            'interaction effects')
    parser.add_argument('-n', '--num', type=int, default=1000, help='sample size '
            'for monte carlo (default: %(default)d)')
    parser.add_argument('-p', '--parameters', help='parameter names',
            type=FileType('r'), metavar='FILE')
    ns = parser.parse_args()
    res = main(ns)

