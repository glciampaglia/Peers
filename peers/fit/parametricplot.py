#!/usr/bin/python

'''
Performs a parametric plot of a GMM sufficient statistics 
'''

import re
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp

from ..utils import fmt

def plot(data):
    pp.close('all')
    f = pp.figure(figsize=(12,4))
    pp.subplot(131)
    add_plot(data[1], data[2], r'$\mu_1$', r'$\mu_2$')
    pp.subplot(132)
    add_plot(data[3], data[4], r'$\sigma_1$', r'$\sigma_2$')
    pp.subplot(133)
    add_plot(data[0], data[5], r'$\varepsilon$', r'$\pi_1$')
    f.subplots_adjust(left=.05, bottom=.12, right=.97, top=.92, wspace=.25)
    pp.show()

def add_plot(x, y, xlab, ylab, **kwargs):
    ax = pp.gca()
    ax.plot(x,y, **kwargs)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    pp.draw()

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data', type=FileType('r'), help='data file')
    parser.add_argument('-m', '--model', metavar='FILE', help='load GP models '
            'from %(metavar)s')
    parser.add_argument('-d', '--delimiter', default=',', help='data fields '
            'delimiter (default: "%(default)s").', metavar='CHAR')
    parser.add_argument('-o', '--output', type=FileType('w'), help='output file')
    return parser

def main(args):
    data = np.loadtxt(args.data, delimiter=args.delimiter).T
    plot(data)
    if args.output is not None:
        pp.savefig(args.output, format=fmt(args.output)) 

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
