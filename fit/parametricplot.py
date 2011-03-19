#!/usr/bin/python

'''
Performs a parametric plot of input data
'''

# TODO <Fri Mar 18 15:22:14 CET 2011> move this functionality into gpfit.py

import re
from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp

def fmt(f):
    return re.search('\.(\w+)$', f.name).group()[1:]

def plot(data):
    pp.close('all')
    f = pp.figure(figsize=(10,4))
    pp.subplot(131)
    add_plot(data[1], data[2], r'$\mu_1$', r'$\mu_2$')
    pp.subplot(132)
    add_plot(data[3], data[4], r'$\sigma_1$', r'$\sigma_2$')
    pp.subplot(133)
    add_plot(data[0], data[5], r'$\varepsilon$', r'$\pi_1$')
#    f.subplots_adjust(left=.05, bottom=.12, right=.97, top=.92, wspace=.25)
    pp.show()

def add_plot(x, y, xlab, ylab, **kwargs):
    ax = pp.gca()
    ax.plot(x,y, **kwargs)
    ax.set_xlabel(xlab, fontsize=14)
    ax.set_ylabel(ylab, fontsize=14)
    pp.draw()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('data', type=FileType('r'), help='data file')
    parser.add_argument('-m', '--model', metavar='FILE', help='load GP models '
            'from %(metavar)s')
    parser.add_argument('-d', '--delimiter', default=',', help='data fields '
            'delimiter (default: "%(default)s").', metavar='CHAR')
    parser.add_argument('-o', '--output', type=FileType('w'), help='output file')
    ns = parser.parse_args()
    data = np.loadtxt(ns.data, delimiter=ns.delimiter).T
    plot(data)
    if ns.output is not None:
        pp.savefig(ns.output, format=fmt(ns.output)) 
