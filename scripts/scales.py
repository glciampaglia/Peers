''' 
estimates time scales tau_0 and tau_1 (long_life and short_life) from empirical
data
'''

from argparse import ArgumentParser
import numpy as np

def lnestim(x):
    '''
    returns mean, median, variance, of lognormal estimated from log-data x
    '''
    m = np.mean(x)
    v = np.var(x, ddof=1)
    return np.exp(m + v / 2), np.exp(m), (np.exp(v) - 1) * np.exp(2 * m + v)

parser = ArgumentParser(description=__doc__)
parser.add_argument('datafn', metavar='data')
parser.add_argument('-output', dest='outfn', metavar='FILE')
parser.add_argument('-log', action='store_true')
parser.add_argument('-batch', action='store_true')
parser.add_argument('-cutoff', type=float, help='default: %(default)d days', 
        default=1)

if __name__ == '__main__':
    ns = parser.parse_args()

    data = np.load(ns.datafn)
    if ns.log:
        data = np.log(data)

    c = np.log(ns.cutoff)

    m0, md0, v0 = lnestim(data[data<c])
    m1, md1 , v1 = lnestim(data[data>c])
    print 'short_life: mean = %g, median = %g, variance = %g' % (m0, md0, v0)
    print 'long_life: mean = %g, median = %g, variance = %g' % (m1, md1, v1)
    if ns.batch:
        import matplotlib
        matplotlib.use('PDF')
    import matplotlib.pyplot as pp
    pp.hist(data, fc='none', bins=20, normed=1)
    pp.axvline(np.log(m0), c='r')
    pp.axvline(np.log(md0), c='b')
    pp.axvline(np.log(m1), c='r')
    pp.axvline(np.log(md1), c='b')

    pp.xlabel('log-days')
    pp.ylabel('density')
    pp.show()
    if ns.outfn:
        pp.savefig(ns.outfn)
