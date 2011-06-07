# coding=utf-8

''' 
Computes confidence intervals for Gaussian Mixture Models using bootstrapping
'''

__author__ = 'Giovanni Luca Ciampaglia'
__email__ = 'ciampagg@usi.ch'

from scikits.learn.mixture import GMM
from multiprocessing import Process, cpu_count, Queue
import numpy as np
from scipy.stats import norm
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

def _fit(model, sample):
    model.fit(sample, n_iter=100)
    means = np.ravel(model.means).copy()
    covars = np.ravel(model.covars).copy()
    weights = np.ravel(model.weights).copy()
    idx = means.argsort()
    return (means[idx], covars[idx], weights[idx])

def _gmmtarget(sample, components=2):
    model = GMM(components)
    return _fit(model, sample)

def _tgmmtarget(sample, components=2):
    model = TGMM(components)
    return _fit(model, sample)

def bootstrap(target, reps, data, size, **kwargs):
    '''
    Computes a statistics by bootstrap
    
    Parameters
    ----------
    target   - user-defined statistics function. Takes in input one data sample.
    reps     - total number of bootstrap samples
    data     - data array
    size     - size of samples passed to target function

    Additional keyword arguments are passed to target function.
    '''
    def _target(queue, data, reps, size, **kwargs):
        ''' executed in each worker process '''
        for sample in bootstrapiter(reps, data, size):
            queue.put(target(sample, **kwargs))
    nprocs = cpu_count()
    if reps % nprocs != 0:
        import warnings
        warnings.warn('rounding down reps from %d to %d' % (reps, 
            (reps / nprocs) * nprocs))
    reps /= nprocs
    queue = Queue()
    args = (queue, data, reps, size)
    pool = [ Process(target=_target, args=args, kwargs=kwargs) 
            for i in xrange(nprocs) ]
    for proc in pool:
        proc.start()
    result = [ queue.get() for i in xrange(reps * nprocs) ]
    for proc in pool:
        proc.join()
    return result

def estimate(x, level=.95):
    ''' 
    Estimate statistics
    
    Parameters
    ----------
    x     - bootstrap statistics
    level - desired confidence level
    '''
    if level > 1 or level < 0:
        raise ValueError('confidence level must be within [0,1]: %g' % level) 
    alpha = norm.ppf(1 - (1 - level) / 2.0) # normal percentile at desired level
    est = np.median(x)
    err = np.std(x, ddof=1) / np.sqrt(len(x))
    ci = alpha * np.std(x, ddof=1)
    return est, err, ci

def main(args):
    data = np.load(args.datafile)
    print
    print 'truncated: %s' % ('yes' if args.truncated else 'no')
    print 'dataset: %s' % args.datafile
    print 'data size: %d observations' % len(data)
    print 'date: %s' % datetime.now()
    print 'size of bootstrap sample: %d' % args.reps
    print 'size of each sample: %d' % (len(data) if args.size is None else 
            args.size)
    print
    if args.truncated:
        result = bootstrap(_tgmmtarget, args.reps, data, args.size,
                components=args.components)
    else:
        result = bootstrap(_gmmtarget, args.reps, data, args.size,
                components=args.components)
    # group by parameter type
    parameters = map(np.asarray, zip(*result)) 
    names = ['μ', 'σ', 'π']
    for i in xrange(args.components):
        print 'component-%d' % (i + 1)
        print '-----------'
        for name, param in zip(names, parameters):
            estim, err, ci = estimate(param[:, i])
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
