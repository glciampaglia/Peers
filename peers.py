#!/usr/bin/env python
# coding=utf-8

# file: peers.py
# vim:ts=8:sw=4:sts=4

''' Pure Python prototype '''

from __future__ import division
from argparse import ArgumentParser
import numpy as np
from collections import deque

class User(object):
    __slots__ = [           # this saves memory
            'opinion',      # \in [0,1], named this way for historic reasons
            'edits',        # \ge 0, number of edits performed
            'successes',    # \ge 0 and \le edits, number of successful edits
            'p_activ',      # \le p_max, probability of activation in dt
    ]
    def __init__(self, args, prng, edits, successes, opinion=None, p_activ=None):
        self.edits = edits
        self.successes = successes
        self.opinion = opinion or prng.rand(0,1)
        self.p_activ = p_activ or prng.rand(0, args.p_max)

class Page(object):
    __slots__ = [
            'opinion',  # see User
            'edits',    # see User
            'p_activ',  # 0 \lt p_activ \le 1, probability of activation in dt
    ]
    def __init__(self, args, prng, edits, opinion=None):
        self.opinion = opinion or prng.rand(0,1)
        self.edits = edits

def update(args, prng, user, page):
    # TODO modify user.opinion and page.opinion
    return user, page

def selection(args, prng, users, pages):
    return deque()

def step_forward(args, prng, users, pages, transient):
    dt = args.time_step
    steps = args.num_transient_steps if transient else args.num_steps
    time = 0.0
    if transient:
        for step in xrange(steps):
            # 1. select (user, page) sequence with updates s.t. each user and each
            # page appear at most once
            # 2. do updates
            pass
    else:
        time = 0.0
        for step in xrange(steps):
            print time, 1, 1
            time += args.time_step
            # see above

def simulate(args):
    prng = np.random.RandomState(args.seed)
    users = [ User(args, prng, 0, 0) for i in xrange(args.num_users) ]
    pages = [ Page(args, prng, .5, 0) for i in xrange(args.num_pages) ] # XXX .5 or None ??
    step_forward(args, prng, users, pages, 1) # don't output anything
    step_forward(args, prng, users, pages, 0) # actual simulation output
    return prng, users, pages

desc = 'The `Peers\' agent-based model © (2010) G.L. Ciampaglia'

def make_parser(): 
    parser = ArgumentParser(description=desc)
    parser.add_argument('-D','--debug', action='store_true', default=False, 
            help='raise Python exceptions to the console')
    parser.add_argument('-s', '--seed', type=int, default=0, 
            help='seed of the pseudo-random numbers generator', metavar='value')
    parser.add_argument('-u', '--num-users', type=int, default=0,
            help='initial number of users', metavar='value')
    parser.add_argument('-p', '--num-pages', type=int, default=0,
            help='initial number of pages', metavar='value')
    parser.add_argument('-t', '--time-step', type=float, default=1/8640, 
            metavar='value', help='Δt update step' )
    parser.add_argument('-n', '--num-steps', type=int, required=True,
            help='number of simulation steps', metavar='value')
    parser.add_argument('-N', '--num-transient-steps', type=int,
            metavar='value', help='number of transient steps', default=0)
    return parser

def print_arguments(args): # print useful info like how many steps to do, etc.
    import sys
    print >> sys.stderr, 'SEED: %d' % args.seed
    print >> sys.stderr, 'INITIAL USERS: %d units' % args.num_users
    print >> sys.stderr, 'INITIAL PAGES: %d units' % args.num_pages
    print >> sys.stderr, 'TIME STEP: %.4e days' % args.time_step
    print >> sys.stderr, 'NUMBER OF STEPS: %d units' % args.num_steps
    print >> sys.stderr, 'TOTAL TIME: %9.10g days' % (args.time_step *
            args.num_steps)
    
def check_arguments(args):
    if args.seed < 0:
        raise ValueError('seed cannot be negative (-s)')
    if args.num_users < 0:
        raise ValueError('num_users cannot be negative (-u)')
    if args.num_pages < 0:
        raise ValueError('num_pages cannot be negative (-p)')
    if args.time_step < 0:
        raise ValueError('time_step cannot be negative (-t)')
    if args.num_steps < 0:
        raise ValueError('num_steps cannot be negative (-n)')

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        check_arguments(ns)
        print_arguments(ns)
        prng, user, pages = simulate(ns)
    except:
        import sys
        ty, val, tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            name = ty.__name__
            print >> sys.stderr, '\n%s: %s\n' % (name, val)
