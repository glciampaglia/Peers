#!/usr/bin/python
# encoding: utf-8

'''Fits simulation data using one or more 1D Gaussian Processes'''

from string import uppercase
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp
from scikits.learn.gaussian_process import GaussianProcess
from cPickle import dump

def getdata(args):
    data = np.loadtxt(args.data, delimiter=args.delimiter)
    N, M = data.shape
    M = M - args.skip_last
    Mi = M - args.number
    return data[:, :Mi], data[:,Mi:M]

def fit(X,y, **kwargs):
    ''' 
    Fits a GP for each column of y. Additional keyword arguments are passed
    to the constructor of scikits.learn.gaussian_processes.GaussianProcess
    '''
    N,M = y.shape
    models = []
    for i in xrange(M):
        gp = GaussianProcess(**kwargs)
        gp.fit(X,y[:,i])
        models.append(gp)
    return models

def rect(x):
    ''' Finds arg min_{m, n <= x and m * n >= x}{ m - n} '''
    x = int(x)
    if x == 1:
        return (1,1)
    if x == 2:
        return (2,1)
    allrows, = np.where(np.mod(x, np.arange(1,x+1)))
    allcols = np.asarray(np.ceil(float(x) / allrows), dtype=int)
    i = np.argmin(np.abs(allrows - allcols))
    return allrows[i], allcols[i]

def plotmodels(x, y, models, innames=None, outnames=None):
    N, M = y.shape
    rows, cols = rect(M)
    fig = pp.figure(figsize=(cols*6, rows*4))
    if M > len(uppercase):
        import warnings
        warnings.warn('not enough labels for this plot (max 26).', 
                category=UserWarning)
    xmin, xmax = x.min(), x.max()
    xi = np.linspace(xmin, xmax, 1000)[:,None]
    for i, label in enumerate(uppercase[:M]):
        gp = models[i]
        ax = pp.subplot(rows, cols, i+1)
        ax.plot(x, y[:,i], 'o ', c='k', label='observations')
        yi, mse = gp.predict(xi, eval_MSE=True)
        ysigma = np.sqrt(mse)
        ax.plot(xi, yi, '-', c='r', label='GP fit')
        pp.fill(np.concatenate([xi, xi[::-1]]), \
                np.concatenate([yi - 1.9600 * ysigma,
                               (yi + 1.9600 * ysigma)[::-1]]), \
                alpha=.5, fc='b', ec='None', label='95% confidence interval')
        pp.axis('tight')
        cymin, cymax = ax.get_ylim()
        ax.set_ylim(cymin, 1.20 * cymax)
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=16,
                fontweight='bold', va='top')
        if innames is not None:
            pp.xlabel(innames[0], fontsize=16)
        else:
            pp.xlabel('model parameter', fontsize=16)
        if outnames is not None:
            pp.ylabel(outnames[i], fontsize=16)
        else:
            pp.ylabel('auxiliary parameter %d' % (i + 1), fontsize=16)
    leg_ax = fig.axes[cols-1]
    # XXX is this placement generic w.r.t. figure size?
    leg_ax.legend(bbox_to_anchor=(0, 1.43, 1.025, 0), loc='upper right')
    fig.subplots_adjust(hspace=.3)
    pp.draw()
    pp.show()

def getnames(namesfile, delimiter):
    if namesfile is not None:
        return namesfile.readline().strip().split(delimiter)

def main(args):
    innames = getnames(args.input_name, args.delimiter)
    outnames = getnames(args.output_name, args.delimiter)
    X, y = getdata(args)
    gpparams = dict(
            theta0 = args.theta0, 
            thetaU = args.thetaU, 
            thetaL = args.thetaL,
            nugget = args.nugget
    )
    models = fit(X,y, **gpparams)
    n, N = X.shape
    if N == 1 and args.plot:
        plotmodels(X, y, models, innames, outnames)
    if args.output:
        dump(models, args.output, 2)

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data', type=FileType('r'), help='input data')
    parser.add_argument('-n', '--number', type=int, help='number of GP models'
            ' to fit')
    parser.add_argument('-d', '--delimiter', default=',', metavar='CHAR',
            help='input fields separator (default: "%(default)s")')
    parser.add_argument('-o', '--output', type=FileType('w'), help='save fitted'
            ' GP models to %(metavar)s', metavar='FILE')
    parser.add_argument('-p', '--plot', help='plot GP fits (if input is 1D)',
            action='store_true')
    parser.add_argument('--input-name', help='input variables names',
            type=FileType('r'), metavar='FILE')
    parser.add_argument('--output-name', help='output variables names',
            type=FileType('r'), metavar='FILE')
    parser.add_argument('--skip-last', help='skip last data column', 
            action='store_true')
    parser.add_argument('-0', '--theta0', type=float, help='GP parameter Theta0'
            ' (default: %(default)g)', metavar='VALUE')
    parser.add_argument('-U', '--thetaU', type=float, help='GP parameter ThetaU'
            ' (default: %(default)g)', metavar='VALUE')
    parser.add_argument('-L', '--thetaL', type=float, help='GP parameter ThetaL'
            ' (default: %(default)g)', metavar='VALUE')
    parser.add_argument('-N', '--nugget', type=float, help='GP parameter nugget'
            ' (default: %(default)g)', metavar='VALUE')
    parser.set_defaults(theta0=.1, thetaL=1e-2, thetaU=1, nugget=1e-2)
    ns = parser.parse_args()
    main(ns)
