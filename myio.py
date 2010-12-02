''' wrappers to numpy.lib.io.load and NpzFile '''

import numpy as np
from numpy.lib.io import NpzFile, load as _load, savez
from itertools import izip, groupby
from warnings import warn

__all__ = [ 'load', 'SimulationsFile' ]

def load(file, mmap_mode='r'):
    ''' 
    if numpy.lib.io.load returns an NpzFile then use the ZIP file's name and
    open a SimulationsFile from it. Else return a NumPy load's result.
    '''
    res = _load(file, mmap_mode=mmap_mode)
    if isinstance(res, NpzFile):
        return SimulationsFile(res.zip.fp.name)
    return res

# XXX
def save(file, name, index, defaults, simulations):
    arcdict = {}
    arcdict['%s_index' % name] = index
    arcdict['%s_defaults' % name] = defaults
    arcdict.update([ ('%s-%d' % (name,i), data) for i, data\
            in enumerate(simulations) ])
    savez(file, **arcdict)

class SimulationsFile(NpzFile):
    '''
    specialization of NpzFile for simulation output. Has additional attributes
    for index, defaults, etc.
    '''
    def __init__(self, fid):
        super(SimulationsFile, self).__init__(fid)
        fn0 = self.files[0]
        if fn0.endswith('_index') or fn0.endswith('_defaults'):
            prefix = fn0.split('_')[:-1] # e.g. "my_sim_index" => "my_sim"
            self.name = '_'.join(prefix)
        elif fn0.find('-'):
            self.name = fn0.split('-')[0] # e.g. "my_sim-0" => "my_sim"
        else:
            raise ValueError('not an archive of simulations')
# other sanity checks
        if self.name + '_index' not in self.files:
            raise ValueError('missing index')
        if self.name + '_defaults' not in self.files:
            warn('missing defaults', category=UserWarning)
            self.miss_defaults = 1
        else:
            self.miss_defaults = 0
        if  len(self) != len(self.index):
            raise ValueError('index/data mismatch')
    def __len__(self):
        return len(self.files) - 2 + self.miss_defaults
    def __iter__(self):
        for i in xrange(len(self)):
            yield self[self.name+'-%d' % i]
    @property
    def index(self):
        return self[self.name + '_index']
    @property
    def indexset(self):
        # NOTE assumes that the index is sorted, otherwise zip(a.indexset,
        # a.itergrouped()) won't match
        raw_index = set(map(tuple, self.index))
        return np.asarray(sorted(raw_index), dtype=self.index.dtype)
    @property
    def defaults(self):
        try:
            return self[self.name + '_defaults']
        except KeyError:
            return []
    def itergrouped(self):
        keyfunc = lambda k : tuple(k[0])
        zipiter = izip(self.index, iter(self))
        for k, subiter in groupby(zipiter, keyfunc):
            yield [data for k, data in subiter ]

