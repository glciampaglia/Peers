''' creates an NPZ archive with simulation data '''

# TODO <Thu Dec  2 12:50:12 CET 2010> add defaults info

import sys
import os
import zipfile
from argparse import ArgumentParser, FileType
import numpy as np
from myio import save

description='''Creates a compressed archive with simulation output data.
The archive contains information on the input parameters to the simulation runs
and can be read using NumPy's I/O function `load`'''

def make_parser():
    parser = ArgumentParser(description=description)
    parser.add_argument('name', help='archive file name')
    parser.add_argument('sample', help='input sample file')
    parser.add_argument('data', nargs='+', help='data files')
    parser.add_argument('-p', '--parameters', nargs='+', help='parameter names',
            required=1)
    parser.add_argument('-d', '--defaults', help='defaults file', metavar='FILE')
    parser.add_argument('-r', '--remove', action='store_true', 
            help='remove input files')
    parser.add_argument('-n', '--notest', action='store_false', dest='test', 
            help='don\'t run the CRC check on the zip file')
    return parser

def make_index(args):
    fn = args.name + '_index.npy'
    N = len(args.parameters)
    named_dtype = np.dtype(zip(args.parameters, [ np.double ] * N ))
    index = np.loadtxt(args.sample, delimiter=',', dtype=named_dtype)
    np.save(fn, index)
    return fn

def make_defaults(args):
    fn = args.name + '_defaults.npy'
    try:
        f = open(args.defaults)
        np.save(fn, f.readlines())
    finally:
        f.close()
    return fn

def main(args):
    args.data.append(make_index(args))
    if args.defaults:
        args.data.append(make_defaults(args))
    args.archive_file = args.name + '.npz'
    zf = zipfile.ZipFile(args.archive_file,'w')
    print 'compressing files ...'
    for f in args.data:
        zf.write(f)
        print '.',
        sys.stdout.flush()
    if args.test:
        zf.testzip()
    if args.remove:
        for f in args.data:
            os.remove(f)
    print '\ndone!' 

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
