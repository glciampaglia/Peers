'''
Test whether EM provides a consistent estimator.

Take a random parameter vector, sample T independent observations from a GMM
with that vector of parameters, fit a GMM to the sample via EM and compute
residuals of the fitted parameters. As T grows, absolute average residuals
should go to 0.
'''

import sys
import numpy as np
from scikits.learn.mixture import GMM
from argparse import ArgumentParser, FileType
import matplotlib.pyplot as pp

def sample_params(args, prng=np.random):
    '''
    Samples parameters for a univariate GMM

    Returns
    -------
    means, uniformly distributed in [-mmax, +mmax)
    std, unformly distributed in (0, smax]
    pi, dirichlet distributed with parameter vector alpha (uniform)
    '''
    shape = (args.reps, args.components)
    m = prng.uniform(-args.max_mean, args.max_mean, size=shape)
    s = - prng.uniform(-args.max_std,0, size=shape) # in this way it excludes 0
    p = prng.dirichlet(args.alpha * np.ones(args.components), size=(args.reps,))
    return m[...,None] ,s[...,None], p

def identify(m,s,p):
    m, s, p = map(np.ravel, (m, s, p))
    idx = m.argsort()
    return m[idx], s[idx], p[idx]

def main(args):
    prng = np.random.RandomState(args.seed)
    T = np.logspace(args.start, args.num + 1, base=args.base, num=args.num)
    print 'sample size: %s' % map(int,T)
    sys.stdout.flush()
    resid = []
    for t in T:
        mresid, sresid, presid = [],[],[]
        for m, s, p in zip(*sample_params(args, prng)):
            gmm = GMM(args.components)
            gmm.means = m
            gmm.covars = s
            gmm.weights = p
            x = gmm.rvs(t)
            gmm2 = GMM(args.components)
            gmm2.fit(x, n_iter=20)
            m, s, p = identify(m, s, p)
            m2, s2, p2 = identify(gmm2.means, gmm2.covars, gmm2.weights)
            mresid.extend(np.abs(m - m2))
            sresid.extend(np.abs(s - s2))
            presid.extend(np.abs(p - p2))
        nsq = np.sqrt(args.reps)
        mresid = (t, np.mean(mresid), np.std(mresid) / nsq)
        sresid = (t, np.mean(sresid), np.std(sresid) / nsq)
        presid = (t, np.mean(presid), np.std(presid) / nsq)
        resid.append((mresid, sresid, presid))
    return np.asarray(zip(*resid))

labels = [r'mean $\mu$', r'st. dev. $\sigma$', r'weights $\pi$']

def plot(args, resid):
    fig = pp.figure(figsize=(11.5,4))
    for i, (data, name) in enumerate(zip(resid, labels)):
        ax = fig.add_subplot(1,3,i+1)
        ax.set_yscale('log')
        ax.set_xscale('log', basex=2)
        ax.errorbar(data.T[0], data.T[1], 0.5 * data.T[2], 
                label=name, marker='o', ls=':')
        ax.legend(loc='upper right')
        ax.set_xticklabels(map(int, resid[0,:,0]))
        ax.set_xlim(data[0,0],data[-1,0])
        ax.set_ylim(0,1)
        if i == 0:
            ax.set_xlabel(r'sample size $T$')
            ax.set_ylabel(r'average abs. residual $|\xi|$')
    fig.subplots_adjust(left=.15, right=.95, bottom=.2, top=.8, wspace=.3)
    pp.draw()
    # XXX produces png instead of pdf!
    if args.output is not None:
        for out in args.output:
            pp.savefig(out)
    pp.show()

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('start', type=float, help='sample size start exponent')
    parser.add_argument('num', type=int, help='number of sample sizes')
    parser.add_argument('components', type=int, help='number of GMM components')
    parser.add_argument('seed', type=int, nargs='?', help='Seed of the PRNG')
    parser.add_argument('-d', '--double', action='store_const', dest='base',
        const=2, default=10)
    parser.add_argument('-r','--reps', type=int, default=100, help='number of '
            'observations for the mean (default: %(default)s)')
    parser.add_argument('-m','--max-mean', default=1, type=float, help='max mean'
            ' hyperparameter (default: %(default)s)')
    parser.add_argument('-s','--max-std', default=2, type=float, help='max'
            ' variance hyperparameter (default: %(default)s)')
    parser.add_argument('-a','--alpha', default=1, type=float, help='weight '
            'hyperparameter alpha (default: %(default)s)')
    parser.add_argument('-o', '--output', type=FileType('w'), help='save '\
            'graphics in %(metavar)s', nargs='+', metavar='FILE')
    ns = parser.parse_args()
    resid = main(ns)
    plot(ns, resid)
