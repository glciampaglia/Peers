''' 
Post-processing script that computes user lifetime. Reads a stream of user/page
interactions and outputs times of first and last edit. 
'''

import sys
from argparse import ArgumentParser, FileType
import numpy as np

def main(args):
    lt = {}
    for line in iter(args.input.readline, ''):
        time, user, page = map(float,line.strip().split())
        try:
            start, stop, edits = lt[user]
            lt[user] = (start, time, edits+1)
        except KeyError:
            lt[user] = (time, -1, 1)
    if len(lt) > 0:
        data = np.asarray(lt.values())
        # filter out users with less edits than min_edits
        data = data[data[:,2] >= args.min_edits]
        if args.lifetime or args.log_lifetime:
            data = np.diff(data[:,:2], axis=1)
            if args.log_lifetime:
                data = np.log(data)
        np.save(args.output_file, data)
    else:
        np.save(args.output_file, [])
    return lt

def make_parser():
    parser = ArgumentParser(fromfile_prefix_chars='@', description=__doc__)
    parser.add_argument('output_file', type=FileType('w'), help='output file '\
            '(- for standard output)', metavar='output')
    parser.add_argument('-i', '--input', type=FileType('r'), default='-',
            help='read data from %(metavar)s (default: standard input).')
    parser.add_argument('-l','--lifetime', action='store_true',
            help='output user lifetime')
    parser.add_argument('-L','--log-lifetime', action='store_true',
            help='output user log-lifetime')
    parser.add_argument('-m', '--min-edits', type=int, default=2, metavar='NUM',
            help='filter out users with less edits than num (default 2)')
    return parser

def check_arguments(args):
    if args.min_edits < 0:
        raise ValueError('invalid value for min_edits (-m/--min-edits)')

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    check_arguments(ns)
    main(ns)
