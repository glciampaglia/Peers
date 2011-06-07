#!/usr/bin/python
# encoding: utf-8

''' Model calibration via indirect inference. Calibration is performed by
minimization of the distance between the sufficient statistic of a gaussian
mixture model (GMM), estimated by means of the EM algorithm on the empirical
data and on simulated data. Gaussian Process approximation of the simulated GMM
statistic is performed to obtain a smooth function of the model parameters. '''

# TODO (in case) standardize sufficient statistic in objective function

import sys
import csv
from datetime import datetime
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp
from scikits.learn.mixture import GMM
from scikits.learn.cross_val import LeaveOneOut
from scipy.optimize import fmin, fmin_l_bfgs_b
from scipy.stats import linregress

from .truncated import TGMM
from .bootstrap import bootstrap, estimate
from ..utils import SurrogateModel, gettxtdata, sanetext, rect, fmt, AppendTupleAction

def _ppdict(d):
    return ', '.join(map(lambda k : '%s : %s' % k, d.items()))

def print_info(args, cv=False):
    print
    print 'Command: %s' % 'cross-validation' if cv else 'fit'
    print 'Data set: %s' % args.datasetname
    print 'Date: %s' % datetime.now()
    print 'Truncated: %s' % ('yes' if args.truncated else 'no')
    print 'GMM Components: %d' % args.components
    print 'GP parameters: %s' % _ppdict(args.gpparams)
    print 'Optimization method: %s' % 'fmin_l_bfgs_b' if args.bounds else 'fmin'
    print
    sys.stdout.flush()

def fitgmm(data, components, truncated=False, n_iter=100):
    '''
    Fits a GMM with given number of components to data. 

    Parameters
    ----------
    data       - data array
    components - number of mixture components
    truncated  - if True, use a TGMM, else a GMM
    n_iter     - maximum number of EM iterations
    '''
    if truncated:
        model = TGMM(components)
    else:
        model = GMM(components)
    model.fit(data, n_iter=n_iter)
    means = model.means.ravel()
    sigmas = np.sqrt(model.covars).ravel()
    weights = model.weights.ravel()
    idx = means.argsort()
    return np.hstack([means[idx], sigmas[idx], weights[idx]])

def fit(args):
    data = np.load(args.data)
    args.datasetname = args.data.name
    print_info(args)
    R = 3 * args.components # number of GMM parameters
    X, Y = gettxtdata(args.simulations, R, delimiter=args.delimiter)
    if args.bootstrap:
        result = bootstrap(_targetfit, args.bootstrap_reps, data,
                args.bootstrap_size, xsim=X, ysim=Y, bounds=args.bounds, 
                components=args.components, **args.gpparams)
        reportfit(args, *zip(*result))
    else:
        theta = fitgmm(data, args.components)
        xopt = _fit(X, Y, theta, **args.gpparams)
        reportfit(args, *xopt)

def reportfit(args, *fitresults):
    if args.bootstrap:
        print 'Bootstrap repetitions: %g' % args.bootstrap_reps
        if args.bootstrap_size:
            print 'Bootstrap sample size: %g' % args.bootstrap_size
        else:
            print 'Bootstrap sample size: same as dataset'
    for x, name in zip(fitresults, args.paramnames):
        if args.bootstrap:
            xest, xerr, xci = estimate(x)
            print '%s : %g +/- %g (95%% ci: %g)' % (name, xest, xerr, xci)
        else:
            print '%s : %.5g' % (name, x)
    print

# wrapper function needed by bootstrap
def _targetfit(sample, xsim, ysim, bounds, components=2, **gpparams):
    theta = fitgmm(sample, components)
    return _fit(xsim, ysim, theta, bounds, **gpparams)

def _fit(X_sim, Y_sim, Y_fit, bounds=None, **gpparams):
    '''
    performs minimization of error between gp fitted with (X_sim,Y_sim) and
    Y_fit. X_sim are parameter values of the simulation model, Y_sim and Y_fit
    are parameters from the auxiliary model. Y_sim are estimated from the
    simulated output data, and Y_fit are estimated from empirical data.

    Parameters
    ----------
    X_sim, Y_sim - from simulation
    Y_fit        - from empirical data
    bounds       - simulation parameter bounds
    
    Additional keyword arguments are passed to the constructor of
    scikits.learn.gaussian_process.GaussianProcess

    Returns
    -------
    X_fit        - simulation parameters that minimize L2 distance between gp
                   approximation of empirical data
    '''
    gp = SurrogateModel.fitGP(X_sim, Y_sim, **gpparams)
    func = lambda x : np.sum((gp(x) - Y_fit) ** 2)
    x0 = X_sim.mean(axis=0)
    if bounds is not None:
        xopt, fopt, d = fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
        return xopt
    else:
        return fmin(func, x0)

def crossval(args):
    args.datasetname = args.simulations.name
    print_info(args, cv=True)
    N = args.components
    R = 3 * N # number of parameters in a GMM with N components
    X, Y = gettxtdata(args.simulations, R, delimiter=args.delimiter)
    cv = []
    for train_index, test_index in LeaveOneOut(len(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        xopt = _fit(X_train, Y_train, Y_test, bounds=args.bounds, **args.gpparams)
        cv.append(zip(X_test.ravel(), xopt))
    cv = np.asarray(zip(*cv)).swapaxes(1,2)
    reportcrossval(args, *cv)
    return cv

def reportcrossval(args, *cvresults):
    h, w = rect(args.parameters)
    fw, fh = pp.rcParams['figure.figsize']
    figsize = h * fh, w * fw
    fig = pp.figure(figsize=figsize)
    for i in xrange(args.parameters):
        x, y = cvresults[i]
        name = args.paramnames[i]
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        print '-' * len(name)
        print name
        print '-' * len(name)
        print 'Slope: %.5g, intercept: %.5g, Error: +/- %.5g' % (slope, intercept,
                std_err)
        print 'R^2: %.5g, P-value: %.5g' % (r_value ** 2, p_value)
        print
        ax = pp.subplot(h,w,i)
#        ax = pp.axes([0.1, 0.1, 0.85, 0.85])
        ax.plot(x, y, ' o', c='white', figure=fig, axes=ax)
        xlim = x.min(), x.max()
        ax.plot(xlim, xlim, 'r-', alpha=.75)
        pp.axis('tight')
        pp.xlabel(r'observed', fontsize='small')
        pp.ylabel(r'estimated', fontsize='small')
        pp.title(sanetext(name), fontsize='small')
        pp.draw()
    fig.subplots_adjust(hspace=.5, wspace=.3)
    if args.output is not None:
        pp.savefig(args.output, format=fmt(args.output.name))
    pp.show()

def main(args):
    if (args.bounds is not None) and args.parameters != len(args.bounds):
        raise ValueError('must specify %d bounds' % args.parameters)
    args.gpparams = dict(
            theta0 = args.theta0, 
            thetaU = args.thetaU, 
            thetaL = args.thetaL,
            nugget = args.nugget,
    )
    # get parameters names
    if args.index is not None:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(args.index.read(1000))
        args.index.seek(0)
        dreader = csv.DictReader(args.index, dialect=dialect)
        args.paramnames = dreader.fieldnames
    else:
        args.paramnames = [ 'parameter \#%d' % i for i in xrange(args.parameters) ]
    args.func(args)

def make_parser():
    # common arguments are put in a parent parser 
    parent = ArgumentParser(add_help=False)
    parent.add_argument('-p', '--parameters', type=int, help='Number of '
            'simulation parameters', metavar='NUM', default=1)
    parent.add_argument('-c', '--components', type=int, help='Number of '
            'GMM components', metavar='NUM', default=2)
    parent.add_argument('-d', '--delimiter', default=',', metavar='CHAR',
            help='input fields separator (default: "%(default)s")')
    parent.add_argument('-0', '--theta0', type=float, help='GP parameter Theta0'
            ' (default: %(default)g)', metavar='VALUE')
    parent.add_argument('-U', '--thetaU', type=float, help='GP parameter ThetaU'
            ' (default: %(default)g)', metavar='VALUE')
    parent.add_argument('-L', '--thetaL', type=float, help='GP parameter ThetaL'
            ' (default: %(default)g)', metavar='VALUE')
    parent.add_argument('-N', '--nugget', type=float, help='GP parameter nugget'
            ' (default: %(default)g)', metavar='VALUE')
    parent.add_argument('-t', '--truncated', action='store_true')
    parent.add_argument('-i', '--index', type=FileType('r'), help='index file')
    parent.add_argument('-b', '--bounds', type=float, nargs=2,
            action=AppendTupleAction, help='simulation parameter bounds')
    parent.set_defaults(theta0=.1, thetaL=1e-2, thetaU=1, nugget=1e-2)
# faster default settings, give worse results
#    parent.set_defaults(theta0=.1, thetaL=None, thetaU=None, nugget=1e-2)
    # main parser with subparsers (see below)
    parser = ArgumentParser(description=__doc__, parents=[parent])
    subparsers = parser.add_subparsers(help='command')
    # subparser for fitting
    parser_fit = subparsers.add_parser('fit')
    parser_fit.add_argument('data', help='empirical data', type=FileType('r'))
    parser_fit.add_argument('simulations', help='GMM parameters estimated from '
            'simulation data', type=FileType('r'))
    parser_fit.add_argument('-B', '--bootstrap', action='store_true', 
            help='compute standard error and 95%% confidence interval with '
            'bootstrap')
    parser_fit.add_argument('-R', '--bootstrap-reps', type=int, default=10000,
            help='bootstrap repetitions (default: %(default)d)', metavar='REPS')
    parser_fit.add_argument('-S', '--bootstrap-size', type=int, help='size of'
            ' bootstrap samples', metavar='SIZE')
    parser_fit.set_defaults(func=fit)
    # subparser for cross-validation
    parser_cv = subparsers.add_parser('crossval')
    parser_cv.add_argument('simulations', help='GMM parameters estimated from '
            'simulation data', type=FileType('r'))
    parser_cv.add_argument('-o', '--output', type=FileType('w'), metavar='FILE',
            help='save figure to %(metavar)s')
    parser_cv.set_defaults(func=crossval)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
