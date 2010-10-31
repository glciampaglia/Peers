''' generates L.H.D. designs, grids, etc etc '''

import sys
import numpy as np
from lhd import lhd as _lhd
from argparse import ArgumentParser, FileType, Action

descr='Generates Jobs'

def dimstoargs(args):
    res = [ '--%s %%(%s)g' % (name, name) for name in args.dimensions ]
    return ' '.join(res)

def makecmdline(args):
    '''
    returns a command line template where dimensions are translated to arguments
    to main script. 

    'python peers.py --confidence %(confidence)g' > output-%(run)d.npy'
    '''
    cmds = []
    script_args = dimstoargs(args)
    script = args.script + ' ' + script_args
    if args.script_defaults:
        cmds.append(script + ' ' + '@' + args.script_defaults)
    else:
        cmds.append(script)
    if args.post:
        for i, post in enumerate(args.post):
            if i < len(args.post_defaults):
                cmds.append(post + ' ' + '@' + args.post_defaults[i])
            else:
                cmds.append(post)
        cmds[-1] += ' ' + args.output_template + '-%(run)d' + '.npy'
        return ' | '.join(cmds)
    else:
        cmds[0] += ' > ' + args.output_template + '-%(run)d' + '.npy'
        return cmds[0]

def lhd(args):
    m = len(args.dimensions)
    prng = np.random.RandomState(args.seed)
    if args.maximin:
        d,design = _lhd(m, args.size, ranges=args.ranges, maximin=1,
                num=args.maximin_trials, prng=prng)
    else:
        d,design = _lhd(m, args.size, ranges=args.ranges, prng=prng)
    return design

def iterjobs(args, design):
    cmdline = makecmdline(args)
    for i, point in enumerate(design):
        argsdict = dict(zip(args.dimensions, point))
        argsdict['run'] = i
        yield (cmdline % argsdict)

def grid(args):
    m = len(args.dimensions)
    design = np.mgrid[[slice(a,b,args.resolution*1j) for a,b in args.ranges]]
    return np.c_[map(np.ravel,design)].T

def make_parser():
    parser = ArgumentParser(description=descr)
    parser.add_argument('script', help='simulation script')
    parser.add_argument('-D', '--debug', action='store_true', 
            help='Raise Python exceptions to the console')
    parser.add_argument('-p', '--post', action='append', default=[], 
            help='add post-processing script', metavar='script')
    parser.add_argument('-d', '--dimensions', action='append',
            help='define a simulation argument', default=[])
    parser.add_argument('-r', '--ranges', nargs=2, type=float, metavar='value',
            help='define a range of values', action='append', default=[])
    parser.add_argument('-s', '--script-defaults', metavar='file')
    parser.add_argument('-S', '--post-defaults', action='append', default=[])
    parser.add_argument('-o','--output-template', default='output',
            help='template for output files')
    subparsers = parser.add_subparsers()
# LHD parser
    parser_lhd = subparsers.add_parser('lhd', help='Latin hypercube design')
    parser_lhd.add_argument('size', type=int, help='design size')
    parser_lhd.add_argument('-m', '--maximin', help='generate maximin design',
            action='store_true')
    parser_lhd.add_argument('-t', '--maximin-trials', type=int, default=10000,
            help='maximizes minimum distance over %(metavar)s trials',
            metavar='NUM')
    parser_lhd.add_argument('--seed', type=int, help='PRNG seed', metavar='seed')
    parser_lhd.set_defaults(design_func=lhd)
# Grid parser
    parser_grid = subparsers.add_parser('grid', help='Dense grid')
    parser_grid.add_argument('resolution', type=int, help='grid resolution')
    parser_grid.set_defaults(design_func=grid)
    return parser

def check_arguments(args):
    if len(args.dimensions) != len(args.ranges):
        raise ValueError('dimensions/ranges mismatch')

def main(args):
    design = args.design_func(args)
    for job in iterjobs(args, design):
        print job
    dty = np.dtype(zip(args.dimensions, [ np.double ]*len(args.dimensions)))
    np.save(args.output_template + '_index.npy', design.view(dty))

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        check_arguments(ns)
        main(ns)
    except:
        ty,val,tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            name = ty.__name__
            print >> sys.stderr, '\n%s : %s\n' % (name, val)

