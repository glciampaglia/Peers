''' Utility functions and classes '''

# TODO move all Action subclasses into a dedicated module (peers.moreactions ?)

import sys
import re
import os.path
import numpy as np
from subprocess import Popen, PIPE
from scikits.learn.gaussian_process import GaussianProcess
from argparse import Action, _AppendAction, ArgumentError

# TODO add check in __init__ that nargs = 2
# TODO make it subclass _AppendAction 
# TODO should raise ArgumentError instead of calling parser.error
# TODO write __doc__
class AppendRange(Action):
    def __call__(self, parser, ns, values, option_string=None):
        a, b = values
        if a > b:
            parser.error('illegal interval: %g %g' % (a, b))
        getattr(ns, self.dest).append((a, b))

# TODO make it subclass _AppendAction
# TODO write __doc__
class AppendTuple(Action):
    def __call__(self, parser, ns, values, option_string=None):
        option = getattr(ns, self.dest)
        if option is None:
            option = [ tuple(values) ]
        else:
            option.append(tuple(values))
        setattr(ns, self.dest, option)

class AppendMaxAction(_AppendAction):
    '''
    Append to an argument up to a specified maximum number of times.

    Pass argument `maxlen` to ArgumentParser.add_argument or it will raise a
    ValueError.
    '''
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None,
                 maxlen=None):
        if int(maxlen) == maxlen and maxlen > 0:
            self.maxlen = maxlen
        else:
            raise ValueError('maxlen parameter must be an integer: %s' % maxlen)
        super(AppendMaxAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar)
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        if len(items) < self.maxlen:
            super(AppendMaxAction, self).__call__(
                    parser, 
                    namespace, 
                    values, 
                    option_string)
        else:
            raise ArgumentError(self, 'this option cannot be specified more than %d'
                    ' times' % self.maxlen)

# TODO should raise ArgumentError instead of calling parser.error
# TODO write __doc__
class CheckDirAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values) or not os.path.isdir(values):
            parser.error('%s is not a directory!' % values)
        setattr(namespace, self.dest, values)

_sanetext_pat = re.compile(r'([^\\])_')
_sanetext_repl = r'\1\_'

def sanetext(s):
    ''' Escapes `_` characters for TeX processing in labels and other texts '''
    return _sanetext_pat.sub(_sanetext_repl, s)

def ttysize():
    '''
    Returns the size of the terminal by calling stty size, or None if the
    system call raise error from the OS or if standard output is not a tty.
    '''
    p = Popen('stty size'.split(), stdout=PIPE, stderr=PIPE)
    try:
        out = p.communicate()[0]
        if out:
            h, w = out.split()
            return int(h), int(w)
        # else standard out is not a tty; in this case 'stty size' returns an
        # empty string.
    except OSError:
        pass

_IDS = {}

class IncIDMixin(object):
    '''
    Mixin for classes with incrementally identified instances
    
    Notes
    -----
    * Classes and their subclasses DO NOT share the id counter.
    * The id of each instance is stored in attribute __id__ and made available
      as a read-only property called `id'.
    '''
    __slots__ = ['__id__']
    def __new__(cls, *args, **kwargs):
        self = super(IncIDMixin, cls).__new__(cls)
        self.__id__ = _IDS.get(cls, 0)
        _IDS[cls] = self.__id__ + 1
        return self
    @property
    def id(self):
        return self.__id__

def gettxtdata(fn, responses, delimiter=',', with_errors=False, **kwargs):
    '''
    Opens a text data file and read data, separating them between
    input variables X and responses Y. 

    Parameters
    ----------
    fn          - file name or file-like object
    responses   - number of response variables
    with_errors - if True, each response variable is followed by its standard
                  errors (returns X,Y,Ye)

    Additional keyword arguments are passed to numpy.loadtxt.
    '''
    kwargs['delimiter'] = delimiter
    data = np.loadtxt(fn, **kwargs)
    if with_errors:
        M = responses * 2
        X = data[:,:-M]
        Y = data[:,-M::2]
        Ye = data[:, -M+1::2]
        return X, Y, Ye
    else:
        X = data[:,:-responses]
        Y = data[:,-responses:]
        return X, Y

def fmt(fn, default=None):
    '''
    Returns the extension of given filename `fn` or `default` if the file sports
    no extension at all.
    '''
    return os.path.splitext(fn)[1][1:] or default

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
        return '<SurrogateModel of %s>' % repr(self.models)

