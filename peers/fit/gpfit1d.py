#!/usr/bin/python
# encoding: utf-8

'''Fits simulation data using one or more 1D Gaussian Processes'''

from string import uppercase
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp
from matplotlib.font_manager import FontProperties
from scikits.learn.gaussian_process import GaussianProcess
from cPickle import dump

from ..utils import gettxtdata, AppendMaxAction, SurrogateModel, rect, fmt

def plotmodels(x, y, model, innames=None, outnames=None, output=None):
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
        gp = model.models[i]
        ax = pp.subplot(rows, cols, i+1)
        ax.plot(x, y[:,i], 'o ', c='k', label='observations')
        yi, mse = gp.predict(xi, eval_MSE=True)
        ysigma = np.sqrt(mse)
        ax.plot(xi, yi, '-', c='r', label='GP fit')
        pp.fill(np.concatenate([xi, xi[::-1]]), \
                np.concatenate([yi - 1.9600 * ysigma,
                               (yi + 1.9600 * ysigma)[::-1]]), \
                alpha=.5, fc='b', ec='None', label='95\% confidence interval')
        pp.axis('tight')
        cymin, cymax = ax.get_ylim()
        ax.set_ylim(cymin, 1.20 * cymax)
        ax.text(0.05, 0.95, r'{\sf %s}' % label, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top')
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
    leg_ax.legend(bbox_to_anchor=(0, 1.43, 1.025, 0), loc='upper right',
            prop=FontProperties(size='small'))
    fig.subplots_adjust(hspace=.3)
    pp.draw()
    if output is not None:
        pp.savefig(output, format=fmt(output.name, 'pdf'))
    pp.show()

def main(args):
    innames, outnames = None, None
    if args.input_name is not None:
        innames = args.input_name.readline().strip().split(args.delimiter)
    if args.output_name is not None:
        outnames = args.output_name.readline().strip().split(args.delimiter)
    X, Y = gettxtdata(args.input, args.responses, delimiter=args.delimiter)
    gpparams = dict(
            theta0 = args.theta0, 
            thetaU = args.thetaU, 
            thetaL = args.thetaL,
            nugget = args.nugget
    )
    model = SurrogateModel.fitGP(X, Y, **gpparams)
    n, N = X.shape
    if N == 1 and args.plot:
        plotmodels(X, Y, model, innames, outnames, args.output)
    elif N > 1:
        raise ValueError('expected univariate input, found N = %d' % N)

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('input', type=FileType('r'), help='input data')
    parser.add_argument('responses', type=int, help='number of response '
            'variables')
    parser.add_argument('-e','--with-errors', action='store_true', 
            help='response variable columns are interleaved with their standard'
            ' errors')
    parser.add_argument('-d', '--delimiter', default=',', metavar='CHAR',
            help='input fields separator (default: "%(default)s")')
    parser.add_argument('-o', '--output', help='save plot to %(metavar)s',
            metavar='FILE', type=FileType('w'))
    parser.add_argument('-p', '--plot', action='store_true', help='plot fit '
            '(univariate input only!)')
    parser.add_argument('--input-name', help='input variables names',
            type=FileType('r'), metavar='FILE')
    parser.add_argument('--output-name', help='output variables names',
            type=FileType('r'), metavar='FILE')
    parser.add_argument('-0', '--theta0', type=float, help='GP parameter Theta0'
            ' (default: %(default)g)', metavar='VALUE')
    parser.add_argument('-U', '--thetaU', type=float, help='GP parameter ThetaU'
            ' (default: %(default)g)', metavar='VALUE')
    parser.add_argument('-L', '--thetaL', type=float, help='GP parameter ThetaL'
            ' (default: %(default)g)', metavar='VALUE')
    parser.add_argument('-N', '--nugget', type=float, help='GP parameter nugget'
            ' (default: %(default)g)', metavar='VALUE')
    parser.set_defaults(theta0=.1, thetaL=1e-2, thetaU=1, nugget=1e-2)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
