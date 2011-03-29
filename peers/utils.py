import sys
from subprocess import Popen, PIPE

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

