import sys
from subprocess import Popen, PIPE

def ttysize():
    '''
    Returns the size of the terminal by calling stty size, or None if the
    system call raise error from the OS.
    '''
    p = Popen('stty size'.split(), stdout=PIPE, stderr=PIPE)
    try:
        return map(int, p.communicate()[0].split())
    except OSError:
        pass

_IDS = {}

class IncIDMixin(object):
    '''
    Mixin for classes with incremental identities. 
    
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

def _myformatwarning(*args):
    msg = args[0]
    return '* WARNING: %s\n' % msg.args[0]

# custom warning formatting
formatwarning = _myformatwarning
import warnings
# default is saved in warnings._formatwarning
warnings._formatwarning = warnings.formatwarning
warnings.formatwarning = formatwarning
