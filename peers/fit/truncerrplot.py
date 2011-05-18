'''plots absolute error of 2-components GMM fit to truncated data'''

import numpy as np
from peers.fit.truncated import TGMM, plot as tgmmplot
from scikits.learn.mixture import GMM
from peers.fit.gmm import plot as gmmplot
from argparse import ArgumentParser
from pylab import *

# defaults
markers = 'od'
components = 2
mu1, mu2 = -1, 1

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-b', '--bounds', nargs=2, type=float, required=1)
    parser.add_argument('-m', '--means', nargs=2, type=float, default=(mu1,
        mu2))
    parser.add_argument('-r', '--reps', type=int, default=10)
    parser.add_argument('sizes', nargs='+', type=int)
    parser.add_argument('-loglog', action='store_true')
    return parser

def main(ns):
    global markers, components
    res = []
    for s in ns.sizes:
        tgmm = TGMM(2, ns.bounds )
        tgmm.means = ns.means
        tgmm.covars = [1,1]
        tgmm.weights = [.5,.5]
        allsamples = tgmm.rvs((ns.reps, s))
        tmp = []
        for sample in allsamples:
            gmm = GMM(components)
            gmm.fit(sample[:,np.newaxis])
            means = gmm.means.ravel()
            means.sort()
            tmp.append(np.abs(means - tgmm.means))
        res.append(zip([s] * components, np.asarray(tmp).mean(axis=0)))
    res = np.asarray(res).swapaxes(0,1)
    for i in xrange(components):
        hold(1)
        x, y = res[i].T
        if ns.loglog:
            loglog(x, y, ':'+markers[i], c='k', label='$\mu_%d$' % (i+1))
        else:
            plot(x, y, ':'+markers[i], c='k', label='$\mu_%d$' % (i+1))
    legend()
    xlabel('sample size')
    ylabel('average absolute error')
    title(r'$\mu_1 = %g,\quad \mu_2 = %g,\quad x\in\left[%g,%g\right]$' % 
            (tuple(ns.means) + tuple(ns.bounds)))
    show()

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)


