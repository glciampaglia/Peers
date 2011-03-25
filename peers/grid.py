'''Regulare lattice input sample generation script'''

import sys
import numpy as np
from argparse import ArgumentParser

description='Regulare lattice input sample generation script'

def make_parser():
    parser = ArgumentParser(description=description)
    parser.add_argument(
            'size', 
            type=int)
    parser.add_argument(
            'dimension',
            type=int)
    parser.add_argument(
            '-i',
            '--intervals',
            nargs=2,
            type=float,
            action='append',
            metavar='VALUE')
    parser.add_argument(
            '-S',
            '--separator',
            default=',',
            help='output separator (default: \'%(default)s\')')
    parser.add_argument(
            '-D',
            '--debug',
            action='store_true',
            help='raise Python exceptions to the console')
    parser.add_argument(
            '-r',
            '--repeat',
            type=int,
            default=1,
            metavar='NUM',
            help='repeat each input point %(metavar)s times')
    return parser

def main(args):
    if args.intervals:
        design = np.mgrid[[slice(a,b,args.size*1j) for a,b in args.intervals]]
    else:
        design = np.mgrid[[slice(0,1,args.size*1j) for i in
            xrange(args.dimension) ]]
    design = np.c_[map(np.ravel,design)].T
    design = np.repeat(design, args.repeat, axis=0)
    np.savetxt(sys.stdout, design, '%g', delimiter=',')

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        main(ns)
    except:
        ty,val,tb = sys.exc_info()
        if ns.debug:
            raise ty,val,tb
        else:
            name = ty.__name__
            parser.error('%s : %s' % (name, val))

