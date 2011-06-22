import numpy as np
import matplotlib.pyplot as pp
from matplotlib import cm
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection

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

def kdeplot(data, xmin=None, xmax=None, num=50, vmin=None, vmax=None, lc='k', **kwargs):
    '''
    Plots density of data, estimated via Gaussian Kernels, together with
    vertical lines for each data point.

    Parameters
    ----------
    data       - data sample
    xmin, xmax - range of density line plot
    num        - number of points at which kde will be evaluated
    vmin, vmax - vertical lines will span from vmin to vmax
    lc         - vertical lines color

    Returns
    -------
    l        - density line object
    linecoll - collection of vertical lines
    kde      - scipy.stats.kde.gaussian_kde

    Additional keyword arguments are passed to plot the density line
    '''
    data = np.ravel(data)
    x0, x1 = data.min(), data.max()
    xmin = xmin or x0
    xmax = xmax or x1
    x = np.linspace(xmin, xmax, num)
    kde = gaussian_kde(data)
    d = kde.evaluate(x)
    if 'axes' in kwargs:
        ax = kwargs['axes']
    elif 'figure' in kwargs:
        ax = kwargs['figure'].axes[-1]
    elif kwargs.pop('hold', False):
        ax = pp.gca()
    else:
        fig = pp.figure()
        ax = fig.add_subplot(111)
    l = ax.plot(x,d, **kwargs)
    y0, y1 = ax.get_ylim()
    vmax = vmax or -(y1 - y0) / 30.
    vmin = vmin or -(y1 - y0) / 10.
    linesiter = ( [(d, vmax), (d, vmin)] for d in data )
    linecoll = LineCollection(linesiter, color=lc)
    ax.add_collection(linecoll)
    ax.set_ylim(y0 - (y1 - vmin)/9., y1)
    pp.draw()
    return l, linecoll, kde
