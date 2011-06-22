''' 
plots densities of data from a compressed archive of NumPy array data files
'''

from argparse import ArgumentParser
import numpy as np
import re

x = re.compile('\w*?(\d+)')
key = lambda k : int(x.match(k).groups()[0])

import matplotlib
matplotlib.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pp
from peers.graphics import kdeplot

parser = ArgumentParser(description=__doc__)
parser.add_argument('datafile', metavar='data')
parser.add_argument('outfile', metavar='output')
parser.add_argument('-every', type=int)

if __name__ == '__main__':
    ns = parser.parse_args()

    sim = np.load(ns.datafile)
    pages = PdfPages(ns.outfile)
    files = sorted(sim.files, key=key)
    sl = slice(None,None,ns.every)
    npages = len(files[sl])

    print 'plotting %d pages from %s to %s' % (npages, ns.datafile, ns.outfile)

    try:
        for i,f in enumerate(files[sl]):
            kdeplot(sim[f], c='r')
            pp.xlabel(r'$\tau$ (log-days)')
            pp.ylabel(r'density')
            pp.title(f)
            pages.savefig()
            print 'page %d of %d: %s' % (i+1, npages, f)
    finally:
        pages.close()
