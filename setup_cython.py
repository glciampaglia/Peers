''' 
Distutils setup script for development purposes. Use it for building Cython
extension modules.
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

# C includes
_I = [ get_include(), 'peers']

setup(
        cmdclass = { 'build_ext' : build_ext },
        ext_modules = [
            Extension("peers.rand", ["peers/rand.pyx"], include_dirs=_I), 
            Extension("peers.cpeers", ["peers/cpeers.pyx"], include_dirs=_I) 
        ]
)

# OLD: compile an extension that makes direct calls to randomkit in NumPy. This
# is supposed to require a copy of Python.pxi in the source tree.

# # Get the include path for NumPy's randomkit extension module.
# import numpy as np
# from os.path import dirname, join
# numpy_root = dirname(np.__file__)
# numpy_random_include = join(numpy_root, 'random')
# numpy_mtrand_include = join(numpy_random_include, 'mtrand')
# 
# # Put this in the list passed to argument ext_modules of setup.
# ext_state = Extension('_state',
#         [ '_state.pyx', 'randomkit.c' ],
#         depends=[ 'randomkit.h', 'Python.pxi' ],
#         include_dirs=[ numpy_include, numpy_random_include ],
#         extra_compile_args=['-Winline',])

