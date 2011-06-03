import numpy as np
import matplotlib.pyplot as pp
from matplotlib import cm

def stackedarea(x, components, weights, cmap=cm.YlGnBu, **kwargs):
    '''
    Produces a stacked area plot from given components and weights.
    
    Parameters
    ----------
    x           - ordinates
    components  - a sequence of objects with a `pdf' method 
    weights     - a sequence of components' weights 

    Default color map is Yellow-Green-Blue. Additional keyword arguments are
    passed matplotlib.pyplot.fill_between. Returns a list of PolyCollections
    (one for each component).
    '''
    assert np.allclose(np.sum(weights), 1) and np.all(weights), 'illegal weights'
    p = [ w * comp.pdf(x) for comp, w in zip(components, weights) ]
    p = [ np.zeros(len(x)) ] + p
    p = np.cumsum(p, axis=0)
    N = len(p)
    colors = cmap(np.linspace(0, 1, N) * (1 - 1.0 / N)) 
    ret = []
    for i in xrange(1, N):
        kwargs['color'] = colors[i-1]
        r = pp.fill_between(x, p[i-1], p[i], **kwargs)
        ret.append(r)
    pp.draw()
    return ret

def mixturehist(data, components, weights, bins=10, num=1000, cmap=cm.YlGnBu, **kwargs):
    '''
    Plots a histogram of given data with a stacked densities of given
    components

    Parameters
    ----------
    data        - data array
    components  - a sequence of random variable objects (see scipy.stats)
    weights     - a sequence of components' weights
    bins        - number of histogram bins
    num         - number of points at which stacked densities are evaluated
    cmap        - stacked area plot color map

    Additional keyword arguments are passed to both matplotlib.pyplot.hist and
    stackedarea.
    '''
    histkw = dict(kwargs)
    # settings for producing transparent histograms
    histkw.update(normed=True, fc=(0,0,0,0), ec='k') 
    pp.hist(data, bins=bins, **histkw)
    xmin, xmax = pp.xlim()
    xi = np.linspace(xmin, xmax, num)
    stackedarea(xi, components, weights, cmap, **kwargs)
