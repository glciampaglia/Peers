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
    parser.add_argument('--edits', action='store_const', dest='func',
            const=edits, help='plot number of edits')
    parser.set_defaults(func=lifetime)
    return parser

def edits(data):
    return data[:,2]

def lifetime(data):
    return np.diff(data[:,:2])

def main(args):
    data = args.func(np.load(args.data))
    if args.log:
        data = np.log(data)
    pp.hist(data, bins=args.bins)
    if args.func is lifetime:
        if args.log:
            xlabel = r'$\log({\rm time})$ (days)'
        else:
            xlabel = r'${\rm time}$ (days)'
    elif args.log:
        xlabel = r'$\log({\rm edits})$'
    else:
        xlabel = r'${\rm edits}$'
    pp.xlabel(xlabel, fontsize=14)
    pp.ylabel(r'${\rm frequency}$', fontsize=14)
    pp.title('sample size = %d' %len(data))
    pp.draw()
    pp.show()

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
