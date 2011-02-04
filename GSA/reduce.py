import os
import sys
import numpy as np
from argparse import ArgumentParser, FileType
from itertools import groupby

def main(args):
    ''' reads index, groups by parameters '''
    if not os.path.exists(args.directory) or not os.path.isdir(args.directory):
        raise ValueError('not a directory: %s' % args.directory)
    liter = ( tuple(l.strip().split(args.sep)) for l in iter(args.index.readline,'') )
    for k, subiter in groupby(liter, lambda k : k[:-1]):
        lt = []
        for fn in ( os.path.join(args.directory, s[-1]) for s in subiter ):
            d = np.load(fn)
            if len(d):
                lt.append(d.mean())
        if len(lt) == 0:
            lt = [ 0. ]
        if args.standard_error:
            out = k + (np.mean(lt), np.std(lt)/np.sqrt(len(lt)))
        else:
            out = k + (np.mean(lt),)
        fmt = ','.join(['%s'] * len(out))
        print fmt % out
        
if __name__ == '__main__':
    parser = ArgumentParser(description='computes average lifetime')
    parser.add_argument('index', type=FileType('r'), help='index file')
    parser.add_argument('-C', '--directory', default='.', help='data directory')
    parser.add_argument('-d', '--delimiter', default=',', help='delimiter of '\
            'index lines', dest='sep')
    parser.add_argument('-s', '--standard-error', action='store_true', 
            help='output standard error values')
    ns = parser.parse_args()
    main(ns)
