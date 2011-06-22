''' plots histogram of input data '''

from os.path import basename, splitext
import numpy as np
from argparse import ArgumentParser
from peers.utils import sanetext

parser = ArgumentParser()
parser.add_argument('datafn', metavar='data')
parser.add_argument('outfn', metavar='output')
parser.add_argument('-log', action='store_true', help='plot log-data')
parser.add_argument('-bins', type=int, default=20, help='default: %(default)d')
parser.add_argument('-batch', action='store_true')
parser.add_argument('-density', dest='normed', action='store_true')

if __name__ == '__main__':
    ns = parser.parse_args()
    if ns.batch:
        import matplotlib
        matplotlib.use('PDF')
    import matplotlib.pyplot as pp

    data = np.load(ns.datafn)
    dataset = basename(splitext(ns.datafn)[0])
    if ns.log:
        data = np.log(data)

    pp.hist(data, bins=ns.bins, fc='none', normed=ns.normed)

    pp.xlabel(r'$\tau$ (log-days)')
    if ns.normed:
        pp.ylabel(r'density')
    else:
        pp.ylabel(r'frequency')
    pp.title(sanetext(dataset))
    pp.draw()
    pp.savefig(ns.outfn)
    pp.show()
