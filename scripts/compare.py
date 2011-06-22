'''
compares empirical data with simulated density
'''

import os.path
from argparse import ArgumentParser
import numpy as np

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as pp

from peers.graphics import kdeplot

parser = ArgumentParser(description=__doc__)
parser.add_argument('datafn', metavar='data')
parser.add_argument('simufn', metavar='simulation')
parser.add_argument('outfn', metavar='output')

if __name__ == '__main__':
    ns = parser.parse_args()
    data = np.load(ns.datafn)
    simu = np.load(ns.simufn)
    dataset = os.path.basename(os.path.splitext(ns.datafn)[0])

    kdeplot(simu, num=100, c='r')
    pp.hist(data, bins=50, fc='none', ec='k', normed=True)
    pp.xlabel(r'$\tau$ (log-days)')
    pp.ylabel(r'density')
    pp.title(dataset)
    pp.savefig(ns.outfn)
    print 'output written to %s' % ns.outfn
