from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as pp
from collections import deque

def moving(x, window):
    if window <= 0:
        raise ValueError('negative window')
    tmp = deque()
    ma  = deque()
    for i in xrange(len(x)):
        ma.append(np.mean(tmp))
        tmp.append(x[i])
        if len(tmp) > window:
            tmp.popleft()
    return ma

def main(args):
    data = np.loadtxt(args.input_file)
    t, users, pages = data.T
    if args.window: 
        users = moving(users, args.window)
        pages = moving(pages, args.window)
    if args.pages and args.users:
        print 'Conflicting arguments. You cannot use both -u/--users and -p/--pages'
        import sys
        sys.exit(-2)
    if args.users:
        pp.plot(t, users, label='users')
    elif args.pages:
        pp.plot(t, pages, label='pages')
    else:
        l,ll = pp.plot(t, users, t, pages)
        l.set_label('users')
        ll.set_label('pages')
    pp.legend(loc=7)
    pp.xlabel('time (days)')
    if args.window:
        pp.title('moving average (window length = %d)' % args.window)
    pp.draw()
    pp.show()

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('input_file', metavar='input', help='input file')
    parser.add_argument('-w','--window', type=int, metavar='length',
            help='plot moving average with given window length')
    parser.add_argument('-p', '--pages', action='store_true', default=False,
            help='plot only number of pages')
    parser.add_argument('-u', '--users', action='store_true', default=False,
            help='plot only number of users')
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
