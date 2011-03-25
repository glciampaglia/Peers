# coding=utf-8
# file: mde.py
# vim:ts=8:sts=4:sw=4

''' Minimum Distance Estimation fitting, with cross-validation '''

import sys
import re
from cStringIO import StringIO
from argparse import ArgumentParser, FileType, Action
import numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial import KDTree
from scipy.stats import ks_2samp, mannwhitneyu

from peers.ioutils import load
from criterion import *

# Kolmogorov-Smirnov, default for continuous data
def ks(a, bseq):
    return [ ks_2samp(a,b)[0] for b in bseq ]

# 2-samples Chi-square distance, for binned data with same number of bins
def chisq(a, bseq):
    return [ chisq_2sam(a, b)[0] for b in bseq ]

# Area under curve
def auc(a, bseq):
    return [ c_auc(a,b) for b in bseq ]

# TODO U is not distance-like. Should find a suitable transformation.
## Mann-Whitney U (or Wilcoxon rank-sum test)
#def mwu(a, bseq):
#    return [ mannwhitneyu(a,b)[0] for b in bseq ]

# Cramér-Von Mises 2 samples test 
def cvm(a, bseq):
    return [ cvmt(a,b) for b in bseq ]

# Anderson-Darling 2-samples test
def adk(a, bseq):
    return [ c_adk([a,b],0) for b in bseq ] # 0 is for non standardized A^2

# estimation of \epsilon
def _epsilon(points, func=np.median):
    ''' computes func (default is numpy.median) of distances over local
    neighborhoods ''' 
    points = np.asarray(points)
    D = points.shape[1]
    tree = KDTree(points[:,:-1])
    distances, _ = tree.query(points[:,:-1],2*D + 1)
    return func(distances[:,1:].ravel())
    
NEED_EPS = ['gaussian', 'multiquadric', 'inverse multiquadric']

def interp_minimize(args):
    '''
    Finds the minimum-distance combination of parameters by interpolation (or
    approximation) of args using scipy.interpolate.Rbf
    '''
    D = len(args.bounds)
    if args.basis in NEED_EPS and args.epsilon is None:
        args.epsilon = _epsilon(args.data)
    r = Rbf(*args.data.T, epsilon=args.epsilon, function=args.basis,
            smooth=args.smooth)
    func = lambda k : r(*k)
    gzm = np.Inf
    X0 = args.prng.rand(args.min_trials, D)
    for x0 in X0:
        xm, zm, infodict = fmin_l_bfgs_b(func, x0, approx_grad=True,
                bounds=args.bounds)
        if zm < gzm:
            gxm = xm
            gzm = zm
            ginfodict = infodict
    return gxm, gzm, ginfodict, r

class FitResult(object):
    ''' place-holder and printing-friendly class for fit results data '''
    __slots__ = [
            'names',
            'theta',
            'min_dist',
            'infodict',
            'rbf'
    ]
    @staticmethod
    def ppdict(keys, values):
        if not keys:
            return ''
        keys = map(str, keys)
        N = np.max(map(len, keys))
        return [ '%s : %s' % (k.rjust(N), v) for k,v in zip(keys,values) ]
    def __str__(self):
        try:
            return self._str()
        except:
            import warnings
            cls_name = self.__class__.__name__
            warnings.warn('problem with %s.__str__' % cls_name,
                    category=UserWarning)
            return object.__repr__(self)
    def _str(self):
        ''' May raise exceptions. Use __repr__ instead. '''
        s = StringIO()
        print >> s, 'theta = (\n%s\n)' % '\n'.join(self.ppdict(self.names,
                self.theta))
        print >> s, 'r(theta) = %g' % self.min_dist
        print >> s, 'status : %(warnflag)d, %(funcalls)d calls, grad = %(grad)s'\
                % self.infodict
        flag = self.infodict['warnflag']
        if flag == 0:
            print >> s, 'Successful.'
        elif flag == 1:
            print >> s, 'Too many evaluations!'
        else:
            print >> s, 'Other reason : %(task)s' % self.infodict
        return s.getvalue()

def fit(args):
    ''' 
    Returns a FitResult instance
    '''
    args.prng = np.random.RandomState(args.seed)
    simulated_data = [ np.vstack(d) for d in args.simulations.itergrouped() ]
    simulated_data = map(np.ravel, simulated_data)
    D = len(args.simulations.index.dtype)
    x = args.simulations.indexset.view('f8').reshape((-1, D))
    z = args.distance(args.data.ravel(), simulated_data)
    args.data = np.c_[ x, z]
    args.bounds = [ ( ax.min(), ax.max() ) for ax in x.T ]
    theta, min_dist, infodict, rbf = interp_minimize(args)
    ret = FitResult()
    ret.theta = theta
    ret.min_dist = min_dist
    ret.infodict = infodict
    ret.rbf = rbf
    ret.names = args.simulations.index.dtype.names
    if args.plot:
        plot_fit_results(args, ret)
    return ret

def plot_fit_results(args, ret):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.cm import jet
    import matplotlib.pyplot as pp
    N = len(args.data) * 2
    u,v = np.mgrid[[slice(b[0],b[1],N*1j) for b in args.bounds]]
    s = ret.rbf(u,v)
    ax = Axes3D(pp.figure())
    if np.all(s > 0):
        from matplotlib.colors import LogNorm
        ax.plot_surface(u,v,s, cmap=jet, norm=LogNorm())
    else:
        ax.plot_surface(u,v,s, cmap=jet)
    ax.plot3D(ret.theta[[0]], ret.theta[[1]],[ ret.min_dist ], 
            marker='o', mfc='w', mec='k', ls='')
    ax.plot3D(args.data.T[0], args.data.T[1], args.data.T[2], 
            mec='w', mfc='k', ls='', marker='o')
    pp.title(r'%s, $s = %g$' % (args.distance.func_name, args.smooth))
    default_labels = ret.names + ( '%s distance' % args.distance.func_name, )
    if args.labels is None:
        args.labels = default_labels
    elif len(args.labels) < len(args.bounds)+1:
        args.labels.extend(default_labels[len(args.labels):])
    setters = [ getattr(ax, 'set_%slabel' % l) for l in 'xyz' ]
    for l,setter in zip(args.labels, setters):
        setter(l, fontsize=16)
    pp.draw()
    pp.show()

class CVResults(object):
    def __init__(self, resdict):
        self.__dict__.update(resdict)
    def __str__(self):
        s = StringIO()
        for name, value in self.__dict__.iteritems():
            r = np.corrcoef(value, rowvar=0)[0,1] 
            rms = np.sqrt(np.add.reduce(np.diff(value)**2))
            print >> s, 'parameter: %s, r = %.2g, RMS = %.2g'\
                    % ( name, round(r,2), round(rms,2))
        return s.getvalue()
    def data(self):
        return dict(**self.__dict__)

def cv(args):
    res = []
    args.prng = np.random.RandomState(args.seed)
    simulated_data = [ np.vstack(d) for d in args.simulations.itergrouped() ]
    simulated_data = map(np.ravel, simulated_data)
    D = len(args.simulations.index.dtype)
    x = args.simulations.indexset.view('f8').reshape((-1, D))
#    z = args.distance(args.data.ravel(), simulated_data)
    args.bounds = [ ( ax.min(), ax.max() ) for ax in x.T ]
    for i in xrange(len(x)):
        data = simulated_data[i]
        reduced_data = list(simulated_data)
        del reduced_data[i]
        reduced_z = args.distance(data, reduced_data)
        reduced_x = list(x)
        del reduced_x[i]
        args.data = np.asarray(np.c_[np.asarray(reduced_x), reduced_z])
        theta, min_dist, infodict, rbf = interp_minimize(args)
        res.append(zip(x[i], theta))
    # group in D 2-dim arrays
    res = map(np.asarray, zip(*res))
    res = CVResults(dict(zip(args.simulations.index.dtype.names, res)))
    if args.output:
        np.savez(args.output, **res.data())
    if args.plot:
        plot_cv_results(args, res.data())
    return res

def plot_cv_results(args, res):
    import matplotlib.pyplot as pp
    for name, points in res.iteritems():
        points = np.asarray(points)
        xlims, ylims = [ ( np.min(pax), np.max(pax) ) for pax in points.T ]
        m, M = points.min(), points.max()
        fig = pp.figure()
        ax = fig.add_subplot(111)
        ax.plot(*points.T, **dict(color='w', marker='o', ls=''))
        ax.plot([m, M], [m, M], color='r')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        tx = xlims[0] + np.diff(xlims) * .25
        ty = ylims[0] + np.diff(ylims) * .75
        ax.text(tx,ty, r'$r = %.2g$' % np.corrcoef(points, rowvar=0)[0,1],
                fontsize=14)
        if name.find('_') is not None:
            ax.set_title(name, fontsize=16)
        else:
            ax.set_title(r'$\%s$' % name,fontsize=16)
        ax.set_xlabel('observed', fontsize=14)
        ax.set_ylabel('predicted', fontsize=14)
    pp.draw()
    pp.show()

DISTANCES = ['ks', 'chisq', 'auc', 'cvm', 'adk']
TARGETS = [ks, chisq, auc, cvm, adk]

BASIS_FUNCS = (
        'multiquadric',
        'inverse',
        'gaussian',
        'linear',
        'cubic',
        'quintic',
        'thin-plate'
)

class MapChoices(Action):
    ''' maps string arguments to target objects '''
    def __init__(self, targets, **kwargs):
        super(MapChoices, self).__init__(**kwargs)
        self.map = dict(zip(self.choices, targets))
    def __call__(self, parser, ns, values, option_string=None):
        try:
            setattr(ns, self.dest, self.map[values])
        except KeyError:
            parser.error('%s is not a valid choice' % values)

class NumPyLoad(Action):
    ''' reads using numpy.lib.io.load '''
    def __init__(self, ext=None, **kwargs):
        super(NumPyLoad, self).__init__(**kwargs)
        self.ext = ext
        self.extpat = r'^.*\.%s$' % ext
    def __call__(self, parser, ns, values, option_string=None):
        if self.ext is not None:
            if not re.match(self.extpat, values, re.I):
                parser.error('wrong extension: %s (expecting: %s)' %
                        (values, self.ext))
        setattr(ns, self.dest, load(values))

description = 'Minimum Distance Estimation tool. © 2010 G.L. Ciampaglia'

def make_parser():
    parser = ArgumentParser(description=description)
    parser.add_argument(
            '-d',
            '--distance',
            metavar='DIST',
            choices=DISTANCES,
            help='%(metavar)s is one of { %(choices)s }, default: %(default)s', 
            default=ks, 
            action=MapChoices,
            targets=TARGETS)
    parser.add_argument(
            '-b',
            '--basis', 
            metavar='FUNC',
            choices=BASIS_FUNCS,
            help='%(metavar)s is one of { %(choices)s }, default: %(default)s', 
            default='multiquadric') 
    parser.add_argument(
            '-s',
            '--smooth', 
            metavar='VAL', 
            type=float,
            help='Rbf smoothness (see scipy.interpolate.Rbf), default: %(default)s',
            default=0)
    parser.add_argument(
            '-e',
            '--epsilon',
            metavar='VAL',
            type=float,
            help='Rbf epsilon (see scipy.interpolate.Rbf) default: estimated from data')
    parser.add_argument(
            '-S', 
            '--seed',
            metavar='SEED',
            help='initialize the random numbers generator with %(metavar)s')
    parser.add_argument(
            '-n', 
            '--min-trials', 
            type=int, 
            default=20, 
            metavar='NUM',
            help='repeat minimization %(metavar)s times (default: %(default)s)')
    parser.add_argument(
            '-D',
            '--debug', 
            action='store_true',
            help='raise Python exceptions to the console')
    subparsers = parser.add_subparsers()
# fit parser
    fit_parser = subparsers.add_parser(
            'fit', 
            help='fit model to data', 
            usage='%(prog)s [OPTIONS] data simulations')
    fit_parser.add_argument(
            'data',
            action=NumPyLoad,
            ext='npy',
            help='data file')
    fit_parser.add_argument(
            '-p',
            '--plot', 
            action='store_true',
            help='plot Rbf distance function (2D only)')
    fit_parser.add_argument(
            '--labels',
            metavar='LABEL',
            nargs='+',
            help='plot labels')
    fit_parser.set_defaults(func=fit)
# cross-validation parser
    cv_parser = subparsers.add_parser(
            'cv',
            help='perform cross validation',
            usage='%(prog)s [OPTIONS] simulations')
    cv_parser.set_defaults(func=cv)
    cv_parser.add_argument(
            '-o',
            '--output', 
            type=FileType('w'), 
            help='write output to %(metavar)s', 
            metavar='FILE')
    cv_parser.add_argument(
            '-p',
            '--plot', 
            action='store_true',
            help='plot cross-validation scatter plots')
    parser.add_argument('simulations',
            help='simulations archive',
            action=NumPyLoad,
            ext='npz')
    return parser

def check_arguments(args, parser):
    if args.min_trials <= 0:
        parser.error('number of trials (-n) must be greater than 0')
    D = len(args.simulations.index.dtype)
    if args.func is fit:
        if args.labels is not None:
            if D + 1 < len(args.labels):
                parser.error('too many labels')

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    check_arguments(ns, parser)
    try:
        ret = ns.func(ns)
        print ret
    except:
        ty,val,tb = sys.exc_info()
        if ns.debug:
            raise ty,val,tb
        else:
            name = ty.__name__
            print >> sys.stderr, '%s : %s' % (name, val)
            sys.exit(1)
