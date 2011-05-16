# coding=utf-8

''' 
Computes confidence intervals for Gaussian Mixture Models using bootstrapping
'''

__author__ = 'Giovanni Luca Ciampaglia'
__email__ = 'ciampagg@usi.ch'

from scikits.learn.mixture import GMM
from multiprocessing import Process, cpu_count, Queue
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

from .truncated import TGMM # should use EM from ctruncated instead

# una cosa interessante sarebbe dividere il sample in subsample e poi ciascun
# processo pesca dal suo sotto sample. Questo permetterebbe di mappare in
# memoria condivisa un dataset grande. Cmq sia il dataset più grande è appena
# 15 mega quindi questo problema non si pone.

def bootstrapiter(n, data, size=None, prng=np.random):
    '''
    Bootstrap sampling iterator. Samples with replacement along the first
    dimension.
    
    Parameters
    ----------
    n    - number of samples
    data - array
    size - if not None, size of each sample, default: len(data)
    prng - RandomState instance
    '''
    x = np.asanyarray(data)
    k = size or len(data) 
    for i in xrange(n):
        idx = prng.randint(0, k, k)
        yield data[idx] 

def _workerfunc(queue, fn, reps, size, components=2, truncated=False):
    data = np.load(fn, mmap_mode='r')
    res = []
    if truncated:
        model = TGMM(components)
    else:
        model = GMM(components)
    for sample in bootstrapiter(reps, data, size):
        model.fit(sample, n_iter=100)
        means = np.ravel(model.means).copy()
        covars = np.ravel(model.covars).copy()
        weights = np.ravel(model.weights).copy()
        idx = means.argsort()
        queue.put((means[idx], covars[idx], weights[idx]))

def main(args):
    nprocs = cpu_count()
    if args.reps % nprocs != 0:
        import warnings
        warnings.warn('rounding down reps from %d to %d' % (args.reps, 
            (args.reps / nprocs) * nprocs))
    args.reps /= nprocs
    queue = Queue()
    a = (queue, args.datafile, args.reps, args.size, args.components,
            args.truncated)
    pool = [ Process(target=_workerfunc, args=a) for i in xrange(nprocs) ]
    for proc in pool:
        proc.start()
    result = [ queue.get() for i in xrange(args.reps * nprocs) ]
    for proc in pool:
        proc.join()
    # group by parameter type
    parameters = map(np.asarray, zip(*result)) 
    print
    print 'date: %s' % datetime.now()
    print 'bootstrap replications: %d' % len(result)
    print
    names = ['μ', 'σ', 'π']
    for i in xrange(args.components):
        print 'component-%d' % (i + 1)
        print '-----------'
        for name, param in zip(names, parameters):
            estim = np.median(param[:,i])
            err = np.std(param[:,i], ddof=1) / np.sqrt(args.reps)
            ci = 1.9600 * np.std(param[:,i], ddof=1)
            print '%s: %g +/- %g (95%% ci: %g)' % (name, estim, err, ci)
        print

def make_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('datafile', help='data file name')
    parser.add_argument('reps', type=int, help='number of models to fit')
    parser.add_argument('-c', '--components', type=int, help='number of mixture'
            ' components (default: %(default)d)', default=2)
    parser.add_argument('-t', '--truncated', action='store_true')
    parser.add_argument('-s', '--size', type=int, help='size of bootstrap '
            'samples')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args() 
    main(ns)
