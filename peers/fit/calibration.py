#!/usr/bin/python
# encoding: utf-8

''' Model calibration via indirect inference. Calibration is performed by
minimization of the distance between the sufficient statistic of a gaussian
mixture model (GMM), estimated by means of the EM algorithm on the empirical
data and on simulated data. Gaussian Process approximation of the simulated GMM
statistic is performed to obtain a smooth function of the model parameters. '''

# TODO <Thu Jun 23 14:39:07 CEST 2011> riportare il valore della funzione di
# errore in corrispondenza del minimo quale misura di GoF della calibrazione.

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
    writer = csv.writer(sys.stdout, dialect=csv.get_dialect('excel'))
    rows = []
    rows.append(('Command', 'cross-validation' if cv else 'fit'))
    rows.append(('Data', args.datasetname))
    rows.append(('Date', datetime.now()))
    rows.append(('Mixture Truncation', ('yes' if args.truncated else 'no')))
    rows.append(('Mixture components', args.components))
    r = ['GP params']
    map(r.extend,args.gpparams.items())
    rows.append(r)
    rows.append(('Optimization method', 
            'fmin_l_bfgs_b' if args.bounds else 'fmin'))
    if cv is False:
        rows.append(('Bootstrap', ('yes' if args.bootstrap else 'no')))
        rows.append(('Bootstrap repetitions', args.bootstrap_reps))
        rows.append(('Bootstrap sample', (args.bootstrap_size if args.bootstrap_size
                else 'same as dataset')))
    if args.weights is not None:
        r = ['Weights']
        map(r.extend, zip(args.auxiliary, args.weights))
        rows.append(r)
    else:
        rows.append(('Weights', 'no'))
    if args.bounds is not None:
        r = ['Bounds']
        map(r.extend, map(lambda k,b : (k,) + b, args.paramnames, args.bounds))
        rows.append(r)
    else:
        rows.append(('Bounds', 'no'))
    writer.writerows(rows)
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
    X, Y = gettxtdata(args.simulations, R, delimiter=args.delimiter, skiprows=1)
    theta = fitgmm(data, args.components)
    estval = _fit(X, Y, theta, bounds=args.bounds, weights=args.weights,
            **args.gpparams)
    if args.bootstrap:
        bootsample = bootstrap(_targetfit, args.bootstrap_reps, data,
                args.bootstrap_size, xsim=X, ysim=Y, bounds=args.bounds, 
                weights=args.weights, components=args.components, 
                **args.gpparams)
        _, esterr, estint = zip(*map(estimate, zip(*bootsample)))
        reportfit(args, estval, error=esterr, interval=estint)
    else:
        reportfit(args, estval)

def reportfit(args, value, error=None, interval=None, level=95):
    fields = ['parameter', 'value', 'error', 'level', 'confint']
    writer = csv.DictWriter(sys.stdout, fields, dialect=csv.get_dialect('excel'))
    writer.writeheader()
    if error is not None:
        for items in zip(args.paramnames, value, error, interval):
            writer.writerow(dict(zip(fields, items)))
    else:
        for name, v in zip(args.paramnames, value):
            row = dict.fromkeys(fields, 'N/A')
            row.update(parameter=name, value=v)
            writer.writerow(row)
    print

# wrapper function needed by bootstrap
def _targetfit(sample, xsim, ysim, bounds, components=2, weights=None, **gpparams):
    theta = fitgmm(sample, components)
    return _fit(xsim, ysim, theta, bounds, weights, **gpparams)

def _fit(thetasim, betasim, beta, bounds=None, weights=None, fmintries=5, **gpparams):
    r'''
    Indirect inference via Gaussian Process approximation. The fitted
    parameters are the solution to the following minization problem::

    $\min_\theta || \beta - \beta_S \left(\theta\right) ||$

    $\beta$ is a vector of parameters of an auxiliary model, estimated on the
    empirical data that we want to fit, $\theta$ is the vector of parameters of
    our simulation model, and $\beta_S\left(\cdot\right)$ is a mapping between
    simulation parameters and auxiliary parameters. $\beta_S$ is substitude with
    a GP approximation, which is estimated using training data
    $\left(\theta^s,\beta^s\right),~~ s=1,\ldots,N$ obtained from simulation.

    Parameters
    ----------
    thetasim     - simulation model parameters (training data)
    betasim      - auxiliary model parameters (training data)
    beta         - auxiliary model parameters estimated on data
    bounds       - list of tuples (a,b) with a<b of simulation parameter bounds
    weights      - weights the contribution of each auxiliary parameter to the
                   error function. Instead of the simple L2 distance, minimize a
                   quadratic form W. The weights parameter can be an array (W =
                   diag(weights)) or define a matrix.
    fmintries    - try minimization multiple times and take best solution
    
    Additional keyword arguments are passed to the constructor of
    scikits.learn.gaussian_process.GaussianProcess

    Returns
    -------
    X_fit        - simulation parameters that minimize L2 distance between gp
                   approximation of empirical data
    '''
    gp = SurrogateModel.fitGP(thetasim, betasim, **gpparams)
    if weights is None:
        weights = np.eye(len(beta))
    weights = np.atleast_1d(weights)
    if weights.ndim == 1:
        weights = np.diag(weights)
    elif weights.ndim == 2:
        N, M = weights.shape
        if N != M:
            raise ValueError('weights must be a square array')
    else:
        raise ValueError('weights can be a 1D or 2D array')
    def func(x):
        d = gp(x) - beta
        return np.dot(np.dot(d, weights), d.T)
    P = thetasim.shape[1]
    range0 = thetasim.ptp(axis=0)
    m0 = thetasim.min(axis=0)
    thetasim.min(axis=0)
    x0 = thetasim.mean(axis=0)
    xopt_best = None
    fopt_best = np.inf
    if bounds is not None:
        for i in xrange(fmintries):
            x0 = range0 * np.random.rand(P) + m0 
            xopt, fopt, d = fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
            if fopt_best > fopt:
                xopt_best = xopt
                fopt_best = fopt
    else:
        for i in xrange(fmintries):
            x0 = range0 * np.random.rand(P) + m0 
            xopt, fopt, calls, flag, vecs = fmin(func, x0, full_output=1, disp=0)
            if fopt_best > fopt:
                xopt_best = xopt
                fopt_best = fopt
    return xopt_best

def crossval(args):
    args.datasetname = args.simulations.name
    print_info(args, cv=True)
    N = args.components
    R = 3 * N # number of parameters in a GMM with N components
    X, Y = gettxtdata(args.simulations, R, delimiter=args.delimiter, skiprows=1)
    cv = []
    for train_index, test_index in LeaveOneOut(len(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        xopt = _fit(X_train, Y_train, Y_test, bounds=args.bounds,
                weights=args.weights, **args.gpparams)
        cv.append(zip(X_test.ravel(), xopt))
    cv = np.asarray(zip(*cv)).swapaxes(1,2)
    reportcrossval(args, *cv)
    return cv

def reportcrossval(args, *cvresults):
    h, w = rect(args.parameters)
    fw, fh = pp.rcParams['figure.figsize']
    figsize = h * fh, w * fw
    fig = pp.figure(figsize=figsize)
    fields = ['parameter','slope', 'intercept', 'error', 'R^2', 'P-value']
    writer = csv.DictWriter(sys.stdout, fields,
            dialect=csv.get_dialect('excel'))
    writer.writeheader()
    for i in xrange(args.parameters):
        name = args.paramnames[i]
        x, y = cvresults[i]
        regr_res = linregress(x, y)
        row = { 'parameter' : name }
        row.update(zip(fields[1:], regr_res))
        writer.writerow(row)
        ax = pp.subplot(h,w,i)
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
    # get field names from input or from simulations file
    if args.fields is not None:
        F = args.parameters + 3 * args.components
        if len(args.fields) != F:
            raise ValueError('expecting %d fields')
        args.paramnames = args.fields[:args.parameters]
        args.auxiliary = args.fields[args.parameters:]
    else:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(args.simulations.readline())
        args.simulations.seek(0)
        dreader = csv.DictReader(args.simulations, dialect=dialect)
        args.paramnames = dreader.fieldnames[:args.parameters]
        args.auxiliary = dreader.fieldnames[args.parameters:]
        args.simulations.seek(0)
    if args.weights is not None:
        a, w = len(args.auxiliary), len(args.weights)
        if a != w:
            raise ValueError('expecting %d weights, not %d' % (a, w))
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
    parent.add_argument('-b', '--bounds', type=float, nargs=2, metavar='VALUE',
            action=AppendTupleAction, help='simulation parameter bounds')
    parent.add_argument('-w', '--weights', type=float, nargs='+', 
            help='auxiliary parameters weights', metavar='VALUE')
    parent.add_argument('-f', '--fields', nargs='+', help='field names')
    parent.set_defaults(theta0=.1, thetaL=1e-2, thetaU=1, nugget=1e-2)
    # main parser inheriting arguments above 
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
