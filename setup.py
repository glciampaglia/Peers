from distutils.core import setup

setup(
        name='peers',
        description='An agent-based simulator for the Peers model.',
        version=0.1,
        author='Giovanni Luca Ciampaglia',
        author_mail='ciampagg@usi.ch',
        packages=[
            'peers', 
            'peers.gsa', 
            'peers.fit', 
            'peers.mde', 
            'peers.tests',
            'peers.design',
        ]
)
