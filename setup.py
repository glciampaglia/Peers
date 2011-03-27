from distutils.core import setup
from distutils.extension import Extension
from numpy import get_include

_I = [ get_include(), 'peers']

setup(
        name='peers',
        description='An agent-based simulator for the Peers model.',
        version=0.0,
        author='Giovanni Luca Ciampaglia',
        author_email='ciampagg@usi.ch',
        packages=[
            'peers', 
            'peers.gsa', 
            'peers.fit', 
            'peers.mde', 
            'peers.tests',
            'peers.design',
        ],
        ext_modules = [
            Extension("peers.rand", ["peers/rand.c"], include_dirs=_I), 
            Extension("peers.cpeers", ["peers/cpeers.c"], include_dirs=_I) 
        ]
)
