''' utility functions '''

import re
import numpy as np
from scikits.learn.gaussian_process import GaussianProcess

def fmt(fn):
    '''
    Returns the extension of given filename fn
    '''
    return re.search('\.(\w+)$', fn).group()[1:]

def rect(x, horizontal=True):
    ''' 
    Finds sides of a rectangle with area at least x, constrained
    to have the smallest difference between width and height. Returns:

       H, W = arg min_{m, n <= x s.t. m * n >= x}{ m - n } 

    Useful for finding the best number of rows and columns for a figure with x
    subplots.

    Parameters
    ----------
    x           - the area or number of subplots
    horizontal  - if True will always return a rectangle w x h with w >= h,
                  viceversa if False

    Returns
    -------
    H, W        - cols x rows of a subplot
    '''
    x = int(x)
    if x == 1:
        return (1,1)
    if x == 2:
        h, w = (1,2)
    else:
        allrows, = np.where(np.mod(x, np.arange(1,x+1)))
        allcols = np.asarray(np.ceil(float(x) / allrows), dtype=int)
        i = np.argmin(np.abs(allrows - allcols))
        h, w = allrows[i], allcols[i]
    if horizontal:
        if w >= h:
            return w, h
        else:
            return h, w
    else:
        if w >= h:
            return h, w
        else:
            return w, h

class SurrogateModel(object):
    '''
    A class that evaluates a gaussian model independently for each response
    variable
    '''
    def __init__(self, models):
        self._models = models
    @property
    def models(self):
        return list(self._models)
    def __call__(self, X, **kwargs):
        '''
        Calls GaussianProcess.predict on each of the GP models of the instance.
        Additional keyword arguments are passed to it. 
        '''
        Y = [ m.predict(X, **kwargs) for m in self._models ]
        return np.row_stack(Y).T
    @classmethod
    def fitGP(cls, X, Y, **kwargs):
        '''
        Fits a gaussian process of X to each variable in Y and returns a
        SurrogateModel instance
        '''
        models = []
        for y in Y.T:
            gp = GaussianProcess(**kwargs)
            gp.fit(X, y)
            models.append(gp)
        return cls(models)
    def __repr__(self):
        return repr(self.models)

