#!/usr/bin/python
'''
compares empirical data with simulated density
'''

import os.path
from argparse import ArgumentParser
import numpy as np

from peers.utils import sanetext

parser = ArgumentParser(description=__doc__)
parser.add_argument('datafn', metavar='data')
parser.add_argument('simufn', metavar='simulation')
parser.add_argument('-output', dest='outfn')
parser.add_argument('-batch', action='store_true')

if __name__ == '__main__':
    ns = parser.parse_args()
    if ns.batch:
        import matplotlib
        matplotlib.use('PDF')

    import matplotlib.pyplot as pp
    from peers.graphics import kdeplot
    
    data = np.load(ns.datafn)
    simu = np.load(ns.simufn)
    dataset = os.path.basename(os.path.splitext(ns.datafn)[0])

    kdeplot(simu, num=100, c='r')
    pp.hist(data, bins=50, fc='none', ec='k', normed=True)
    pp.xlabel(r'$\tau$ (log-days)')
    pp.ylabel(r'density')
    pp.title(sanetext(dataset))
    if ns.outfn:
        pp.savefig(ns.outfn, format='pdf')
        print 'output written to %s' % ns.outfn
    pp.show()
