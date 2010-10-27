#!/usr/bin/env python
import sys
from argparse import ArgumentParser, FileType
import numpy as np

def main(args):
    lt = {}
    for line in iter(sys.stdin.readline, ''):
        time, user, page = map(float,line.strip().split())
        try:
            start, stop = lt[user]
            lt[user] = (start, time)
        except KeyError:
            lt[user] = (time, -1)
    data = np.asarray(lt.values())
    # filter out users with only one edit
    data = data[data[:,1] > 0]
    np.save(args.output_file, data)
    return lt

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('output_file', type=FileType('w'), help='output file', metavar='output')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
