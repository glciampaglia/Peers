from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

#import numpy as np
#from os.path import dirname, join
#numpy_root = dirname(np.__file__)
#numpy_random_include = join(numpy_root, 'random')
#numpy_mtrand_include = join(numpy_random_include, 'mtrand')

#ext_state = Extension('_state',
#        [ '_state.pyx', 'randomkit.c' ],
#        depends=[ 'randomkit.h', 'Python.pxi' ],
#        include_dirs=[ numpy_include, numpy_random_include ],
#        extra_compile_args=['-Winline',])

# the current directory is needed since we are in a package and Cython will try
# to lookup definitions for module rand (rand.pxd) by looking at peers.rand and
# not just for peers.
_includes = [ get_include(), '.']

setup(
        cmdclass = { 'build_ext' : build_ext },
        ext_modules = [
            Extension("rand", ["rand.pyx"], include_dirs=_includes), 
            Extension("cpeers", ["cpeers.pyx"], include_dirs=_includes) 
        ]
)
