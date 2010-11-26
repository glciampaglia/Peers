''' generates synthetic simulated data '''

import re
import sys
import numpy as np
import matplotlib.pyplot as pp
from argparse import ArgumentParser, Action, FileType
from scipy.stats import distributions
from subprocess import Popen

class GetAttr(Action):
    def __init__(self, instance, **kwargs):
        super(GetAttr, self).__init__(**kwargs)
        self.instance = instance
    def __call__(self, parser, ns, values, option_string=None):
        try:
            instance_value = getattr(self.instance, values)
        except AttributeError:
            parser.error('%s is not a valid choice' % values)
        else:
            setattr(ns, self.dest, instance_value)

description = '''reads a collection of parameter points from input and generates
synthetic simulated data using SciPy distribution'''

class FileTypeExt(Action):
    ''' checks extension '''
    def __init__(self, ext=None, mode='r', **kwargs):
        super(FileTypeExt, self).__init__(**kwargs)
        self.ext = ext
        self.mode = mode
        self.extpat = r'^.*\.%s$' % ext
    def __call__(self, parser, ns, values, option_string=None):
        if values == '-':
            setattr(ns, self.dest, sys.stdout)
            return
        if self.ext is not None:
            if not re.match(self.extpat, values, re.I):
                parser.error('wrong extension: %s (expecting: %s)' %
                        (values, self.ext))
        setattr(ns, self.dest, open(values, mode=self.mode))

def make_parser():
    parser = ArgumentParser(description=description)
    parser.add_argument(
            'distribution', 
            action=GetAttr, 
            metavar='DIST',
            help='use DIST from scipy.stats.distributions',
            instance=distributions)
    parser.add_argument(
            'size',
            metavar='SIZE',
            type=int,
            help='sample size',)
    parser.add_argument(
            'output',
            metavar='FILE',
            action=FileTypeExt,
            mode='w',
            ext='npz',
            help='write output to %(metavar)s.')
    parser.add_argument(
            '-r',
            '--repetitions',
            type=int,
            default=1)
    parser.add_argument(
            '-D',
            '--debug',
            action='store_true',
            help='raise Python exceptions to the console')
    parser.add_argument(
            '-i',
            '--input',
            dest='input',
            default=sys.stdin,
            metavar='FILE',
            help='read parameters from %(metavar)s',
            type=FileType('r'))
    return parser

def main(args):
    points = [ tuple(map(float, line.strip().split(','))) for line in args.input]
    points = sorted(points * args.repetitions)
    index = []
    data = []
    for i,p in enumerate(points):
        rvs = args.distribution.rvs(*p, size=args.size)
        index.append(tuple(p))
        data.append(('test-%d' %i, rvs))
    defaults=np.array(['TEST: %s' % args.distribution])
    dty = np.dtype([ ('%s-%d' % (args.distribution.name, i), np.double) for i in xrange(len(p)) ])
    index = np.asarray(index, dtype=dty)
    np.savez(args.output, test_index=np.asarray(index), test_defaults=defaults, 
            **dict(data))

if __name__ == '__main__':
    parser = make_parser()
    try:
        ns = parser.parse_args()
    except:
        ty, val, tb = sys.exc_info()
        # manually check the command line
        if '-D' in sys.argv or '--debug' in sys.argv:
            raise ty, val, tb
        else:
            parser.error('parsing arguments: %s' % val)
    try:
        main(ns)
    except:
        ty, val, tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            parser.error('%s : %s' % (ty.__name__, val))
