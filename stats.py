''' plots histogram of input data '''

import numpy as np
import matplotlib.pyplot as pp
from argparse import ArgumentParser, FileType

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('data', metavar='file', type=FileType('r'),
            help='data file')
    parser.add_argument('-l', '--log', default=False, action='store_true',
            help='take logarithm of the data')
    parser.add_argument('-b', '--bins', type=int, help='number of bins',
            default=10)
    return parser

def main(args):
    data = np.load(args.data)
    if args.log:
        data = np.log(data)
    pp.hist(data, bins=args.bins)
    pp.xlabel('time (days)')
    pp.draw()
    pp.show()

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
