#!/usr/bin/python
# encoding: utf-8

''' Model calibration via indirect inference. Calibration is performed by
minimization of the distance between the sufficient statistic of a gaussian
mixture model (GMM), estimated by means of the EM algorithm on the empirical
data and on simulated data. Gaussian Process approximation of the simulated GMM
statistic is performed to obtain a smooth function of the model parameters. '''

# TODO (in case) standardize sufficient statistic in objective function
# TODO aggiungi standard error calcolato con bootstrap

import csv
from datetime import datetime
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp
from scikits.learn.mixture import GMM
from scikits.learn.cross_val import LeaveOneOut
from scipy.optimize import fmin
from scipy.stats import linregress

from .truncated import TGMM
from ..utils import SurrogateModel, gettxtdata, sanetext, rect, fmt

def _ppdict(d):
    return ', '.join(map(lambda k : '%s : %s' % k, d.items()))

def print_info(args, cv=False):
    print
    print 'Command: %s' % 'cross-validation' if cv else 'fit'
    print 'Data set: %s' % args.datasetname
    print 'Date: %s' % datetime.now()
    print 'Truncated: %s' % 'yes' if args.truncated else 'no'
    print 'GMM Components: %d' % args.components
    print 'GP parameters: %s' % _ppdict(args.gpparams)
    print

def fitgmm(data, components, truncated=False):
    '''
    Fits a GMM with given number of components to data. Additional keyword
    arguments are passed to the constructor of scikits.learn.mixture.GMM.fit
    '''
    if truncated:
        model = TGMM(components)
    else:
        model = GMM(components)
    model.fit(data, n_iter=100)
    means = model.means.ravel()
    sigmas = np.sqrt(model.covars).ravel()
    weights = model.weights.ravel()
    idx = means.argsort()
    return np.hstack([means[idx], sigmas[idx], weights[idx]])

def fit(args):
    data = np.load(args.data)
    theta = fitgmm(data, args.components)
    X, Y = gettxtdata(args.simulations, len(theta), delimiter=args.delimiter)
    xopt = _fit(X, Y, theta, **args.gpparams)
    args.datasetname = args.data.name
    print_info(args)
    for x, name in zip(xopt, args.paramnames):
        print '%s : %.5g' % (x, name)
    print xopt

def _fit(X_sim, Y_sim, Y_fit, **gpparams):
    '''
    performs minimization of error between gp fitted with (X_sim,Y_sim) and
    Y_fit. X_sim are parameter values of the simulation model, Y_sim and Y_fit
    are parameters from the auxiliary model. Y_sim are estimated from the
    simulated output data, and Y_fit are estimated from empirical data.

    Parameters
    ----------
    X_sim, Y_sim - from simulation
    Y_fit        - from empirical data
    
    Additional keyword arguments are passed to the constructor of
    scikits.learn.gaussian_process.GaussianProcess

    Returns
    -------
    X_fit        - simulation parameters that minimize L2 distance between gp
                   approximation of empirical data
    '''
    gp = SurrogateModel.fitGP(X_sim, Y_sim, **gpparams)
    func = lambda x : np.sum((gp(x) - Y_fit)**2)
    x0 = X_sim.mean(axis=0)
    return fmin(func, x0)

def cross_val(args):
    N = args.components
    R = 3 * N # number of parameters in a GMM with N components
    X, Y = gettxtdata(args.simulations, R, delimiter=args.delimiter)
    cv = []
    for train_index, test_index in LeaveOneOut(len(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        xopt = _fit(X_train, Y_train, Y_test, **args.gpparams)
        cv.append(zip(X_test.ravel(), xopt))
    cv = np.asarray(zip(*cv)).swapaxes(1,2)
    args.datasetname = args.simulations.name
    report_results(args, *cv)
    return cv

def report_results(args, *cvresults):
    print_info(args, cv=True)
    h, w = rect(args.parameters)
    fig = pp.figure()
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
        args.paramnames = [ 'parameter #%d' % i for i in xrange(args.parameters) ]
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
    parser_fit.set_defaults(func=fit)
    # subparser for cross-validation
    parser_cv = subparsers.add_parser('crossval')
    parser_cv.add_argument('simulations', help='GMM parameters estimated from '
            'simulation data', type=FileType('r'))
    parser_cv.add_argument('-o', '--output', type=FileType('w'), metavar='FILE',
            help='save figure to %(metavar)s')
    parser_cv.set_defaults(func=cross_val)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
