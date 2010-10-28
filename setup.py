from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from numpy import get_include
numpy_include = get_include()

ext_rand = Extension("rand",
        [ "rand.pyx" ],
        include_dirs=[ numpy_include ])

ext_cpeers = Extension("_cpeers",
        [ "cpeers.pyx" ],
        include_dirs=[ numpy_include, '.'])

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

setup(
        cmdclass = dict(build_ext=build_ext),
        ext_modules = [ ext_rand, ext_cpeers ]
)
