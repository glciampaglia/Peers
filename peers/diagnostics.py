''' plots the number of users and pages '''

from argparse import ArgumentParser, FileType

# TODO <ven 29 ott 2010, 11.34.15, CEST> use scikits
import numpy as np
import matplotlib.pyplot as pp
from collections import deque
import sys

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

def plot_data(args):
    for i,data_file in enumerate(args.input_file):
        print 'processing %s...' % data_file.name,
        sys.stdout.flush()
        data = np.loadtxt(data_file)
        t, users, pages = data.T
        if args.window: 
            users = moving(users, args.window)
            pages = moving(pages, args.window)
        if args.users:
            pp.plot(t, users, label='users #%d' % i, hold=1)
        elif args.pages:
            pp.plot(t, pages, label='pages #%d' %i, hold=1)
        else:
            l,ll = pp.plot(t, users, t, pages)
            l.set_label('users #%d' % i)
            ll.set_label('pages #%d' % i)
        print ' done'
        sys.stdout.flush()

def main(args):
    if args.pages and args.users:
        print 'Conflicting arguments. You cannot use both -u/--users and -p/--pages'
        import sys
        sys.exit(-2)
    plot_data(args)
    pp.legend(loc=7)
    pp.xlabel('time (days)')
    if args.window:
        pp.title('moving average (window length = %d)' % args.window)
    pp.draw()
    pp.show()

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('input_file', metavar='input', help='input file',
            nargs='+', type=FileType('r'))
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
