#!/usr/bin/python

''' 
----------------------------------------------------
Indirect inference fit with GMM sufficient statistic
----------------------------------------------------

This program performs the parameter estimation of the Peers model via an
indirect inference procedure. The statistical feature used for comparison is the
sufficient statistic of a gaussian mixture model (GMM). The objective function
to minimize is the distance between the parameters of a GMM estimated on
empirical data and the parameters of a GMM estimated from the synthetic obtained
from simulation. Since simulation is expensive, instead of using directly the
Peers model, we use a surrogate model of the GMM parameters of the simulated
data, that is, a function mapping values of the parameters of the Peers model to
value of the sufficient statistic of a GMM -- the GMM parameter obtained from
estimation via the EM algorithm.
'''

# TODO <lun 21 mar 2011, 19.02.24, CET>:
# 1. Check GaussianProcess parameters
# 2. Also try cubic splines
# 3. Standardize sufficient statistic in objective function
# 4. Try also fmin or other optimizers?

from argparse import ArgumentParser, FileType
import numpy as np
import matplotlib.pyplot as pp
from scikits.learn.gaussian_process import GaussianProcess
from scikits.learn.mixture import GMM
from scikits.learn.cross_val import LeaveOneOut
from scipy.optimize import leastsq, fmin

class Surrogate(object):
    '''
    Class for surrogate models as callable instance
    '''
    def __init__(self, models):
        self._models = models
    @property
    def models(self):
        return list(self._models)
    def __call__(self, x):
        return np.ravel([ m.predict(x) for m in self._models ])

def fitgp(X,Y, **kwargs):
    ''' 
    Fits a GP for each column of Y. Additional keyword arguments are passed
    to the constructor of scikits.learn.gaussian_processes.GaussianProcess
    '''
    N,M = Y.shape
    models = []
    for i in xrange(M):
        gp = GaussianProcess(**kwargs)
        gp.fit(X,Y[:,i])
        models.append(gp)
    return Surrogate(models)

def fitgmm(data, components, **kwargs):
    '''
    Fits a GMM with given number of components to data. Additional keyword
    arguments are passed to the constructor of scikits.learn.mixture.GMM.fit
    '''
    gmm = GMM(components)
    gmm.fit(data, **kwargs)
    mu, sigma, weights = map(np.ravel, 
            [gmm.means, np.asarray(gmm.covars), gmm.weights])
    idx = mu.argsort()
    params = np.empty(components*3 - 1)
    params[:components] = mu[idx]
    params[components:2*components] = sigma[idx]
    params[2*components:] = weights[idx][:-1]
    return params

def fit(args):
    data = np.load(args.data)
    theta = fitgmm(data, args.components)
    simulations = np.loadtxt(args.simulations, delimiter=args.delimiter)
    X = simulations[:,:args.parameters]
    Y = simulations[:,args.parameters:-1]
    gp = fitgp(X,Y)
    func = lambda x : gp(x) - theta
    x0 = X.mean(axis=0)
    plsq, ier = leastsq(func, x0)
    print plsq

def cross_val(args):
    simulations = np.loadtxt(args.simulations, delimiter=args.delimiter)
    X = simulations[:,:args.parameters]
    Y = simulations[:,args.parameters:-1]
    cv_result = []
    for train_index, test_index in LeaveOneOut(len(simulations)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        gp = fitgp(X_train, Y_train)
        Y_test = Y_test
        func = lambda x : gp(x) - Y_test
        x0 = X_train.mean(axis=0)
        plsq, ier = fmin(func, x0)
        print plsq
        cv_result.append((X_test.item(), plsq))
    cv_result = np.asarray(cv_result)
    pp.scatter(cv_result.T[0], cv_result.T[1])
    pp.draw()
    pp.show()

def main(args):
    if args.cross_val:
        cross_val(args)
    else:
        fit(args)

if __name__ == '__main__':
    parser = ArgumentParser(description='Indirect inference fit with GMM '
            'sufficient statistic')
    parser.add_argument('data', help='empirical data', type=FileType('r'))
    parser.add_argument('simulations', help='GMM parameters estimated from '
            'simulation data', type=FileType('r'))
    parser.add_argument('-p', '--parameters', type=int, help='Number of '
            'simulation parameters', metavar='NUM', default=1)
    parser.add_argument('-c', '--components', type=int, help='Number of '
            'GMM components', metavar='NUM', default=2)
    parser.add_argument('-d', '--delimiter', default=',', metavar='CHAR',
            help='input fields separator (default: "%(default)s")')
    parser.add_argument('-C', '--cross-val', help='perform cross-validation',
            action='store_true')
    ns = parser.parse_args()
    main(ns)
