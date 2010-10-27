#!/usr/bin/env python
# coding=utf-8

# file: peers.py
# vim:ts=8:sw=4:sts=4

''' Pure Python version '''

from __future__ import division
from argparse import ArgumentParser, FileType
import numpy as np
from collections import deque
import sys
from time import time

from rand import randwpmf

# custom warning formatting
formatwarning = lambda msg,cat,fn,lno,l: '*** WARNING *** ' + msg.args[0] + '\n'
import warnings
# default is saved in warnings._formatwarning
warnings._formatwarning = warnings.formatwarning
warnings.formatwarning = formatwarning
from warnings import warn

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
        self.opinion = opinion or prng.rand()               # ~ U[0,1]
        self.p_activ = p_activ or prng.rand() * args.p_max  # ~ U[0,p_max]
    @property
    def p_leave(self):
        den = self.edits
        num = self.successes
        assert num <= den, "user will never stop"
        if den:
            return num / den
        else:
            return 1.

class Page(object):
    __slots__ = [
            'opinion',  # see User
            'edits',    # see User
    ]
    def __init__(self, args, prng, edits, opinion=None):
        self.opinion = opinion
        self.edits = edits

def interaction(args, prng, users, pages, pairs, update_opinions=True):
    for i, j in pairs:
        u = users[i]
        p = pages[j]
        u.edits += 1
        p.edits += 1
        if update_opinions:
            if p.opinion is None: # first edit of page
                p.opinion = u.opinion
                u.successes += 1
            else: # subsequent edits of page
                ok = np.abs(u.opinion - p.opinion) < args.confidence
                if ok:
                    u.successes += 1
                    u.opinion += args.speed * ( p.opinion - u.opinion )
                    p.opinion += args.speed * ( u.opinion - p.opinion )
                elif prng.rand() < args.rollback_prob:
                    p.opinion += args.speed * ( u.opinion - p.opinion )
            print args.time, i, j
            args.noedits += 1
        users[i] = u
        pages[j] = p

def selection(args, prng, users, pages):
    '''
    Returns a sequence of pairs (user, page) for interactions. 
    '''
    if len(pages) == 0:
        return []
    rvs = prng.rand(len(users))
    editing_users = deque()
    for i in xrange(len(users)): # activate user in Δt with prob p_activ
        if rvs[i] <= users[i].p_activ:
            editing_users.append(i)
    # pages are drawn with prob. proportional to popularity (i.e. # of edits)
    page_pmf = np.asarray([ p.edits for p in pages], dtype=np.double)
    editing_pages = randwpmf(page_pmf, size=(len(editing_users),), prng=prng)
    # if a page occurs more than once, only the first user edits it
    already_edited = []
    res = deque()
    for u,p in zip(editing_users, editing_pages):
        if p not in already_edited:
            res.append((u,p))
        already_edited.append(p)
    return res

def update(args, prng, users, pages):
    # NOTE < τ > = Δt * p-stop**-1 ==> p-stop = Δt / < τ >
    rvs = prng.rand(len(users))
    removed = 0
    for i in xrange(len(users)):
        idx = i - removed # deleting shrink the list so need to update idx
        if rvs[i] <= args.p_stop * users[idx].p_leave ** args.stop_exp:
            del users[idx]
            removed += 1
    users.extend([ User(args, prng, args.const_succ, args.const_succ) for i in 
            xrange(prng.poisson(args.user_input_rate)) ])
    pages.extend([ Page(args, prng, args.const_pop) for i in
            xrange(prng.poisson(args.page_input_rate)) ])
    if args.info_file:
        info = {}
        info['time'] = args.time
        info['users'] = len(users)
        info['pages'] = len(pages)
        args.info_file.write('%(time)s %(users)s %(pages)s\n' % info)

def step_forward(args, prng, users, pages, transient):
    dt = args.time_step
    if transient:
        steps = args.num_transient_steps 
    else:
        steps = args.num_steps
    for step in xrange(steps):
        pairs = selection(args, prng, users, pages)
        interaction(args, prng, users, pages, pairs, update_opinions=transient)
        update(args, prng, users, pages)
        args.time += args.time_step

def simulate(args):
    prng = np.random.RandomState(args.seed)
    user_op = prng.rand(args.num_users)
    page_op = prng.rand(args.num_pages)
    users = [ User(args, prng, args.const_succ, args.const_succ, user_op[i]) 
            for i in xrange(args.num_users) ]
    pages = [ Page(args, prng, args.const_pop, page_op[i]) for i in 
            xrange(args.num_pages) ] 
    args.time = 0.0
    step_forward(args, prng, users, pages, 1) # don't output anything
    args.noedits = 0
    start_time = time()
    step_forward(args, prng, users, pages, 0) # actual simulation output
    speed = args.noedits / (time() - start_time) 
    print >> sys.stderr, " *** Speed: %g (interactions/sec)" % speed
    return prng, users, pages

desc = 'The `Peers\' agent-based model © (2010) G.L. Ciampaglia'
usage = '%(prog)s [OPTIONS] duration [seed]'

def make_parser(): 
    parser = ArgumentParser(description=desc, usage=usage)
    #
    # positional arguments
    parser.add_argument('num_steps', type=int, 
            help='number of simulation steps', metavar='duration')
    parser.add_argument('seed', type=int, nargs='?',
            help='seed of the pseudo-random numbers generator', metavar='seed')
    #
    # optional arguments
    parser.add_argument('-i', '--info_file', type=FileType('w'), default=False,
            help='write simulation info to file', metavar='file')
    parser.add_argument('-d', '--dry-run', action='store_true', default=False,
            help='do not simulate, just print parameters defaults')
    parser.add_argument('-D','--debug', action='store_true', default=False, 
            help='raise Python exceptions to the console')
    parser.add_argument('--profile', action='store_true', default=False,
            help='run profiling')
    parser.add_argument('--profile-file', metavar='file', default=None,
            help="store profiling information in file")
    parser.add_argument('-u', '--num-users', type=int, default=0,
            help='initial number of users', metavar='value')
    parser.add_argument('-p', '--num-pages', type=int, default=0,
            help='initial number of pages', metavar='value')
    parser.add_argument('-U', '--user-input-rate', metavar='rate', default=1.0, 
            type=np.double, help='rate of new users per unit of time Δt')
    parser.add_argument('-P', '--page-input-rate', metavar='rate', type=np.double,
            default=1.0, help='rate of new pages per unit of time Δt')
    parser.add_argument('-t', '--time-step', type=np.double, default=1/8640, 
            metavar='value', help='Δt update step' )
    parser.add_argument('-c', '--confidence', metavar='value',
            type=np.double, default=.2, help='confidence parameter')
    parser.add_argument('-s','--speed', metavar='value', type=np.double,
            default=0.5, help='opinion averaging speed')
    parser.add_argument('--transient', dest='num_transient_steps', type=int,
            metavar='value', help='number of transient steps', default=0)
    parser.add_argument('--p-max', metavar='prob', type=np.double,
            default=1, help='user interaction maximum probability') 
    parser.add_argument('--p-stop', metavar='prob', type=np.double,
            default=1e-3, help='user withdrawl baseline probability')
    parser.add_argument('--const-succ', metavar='value', type=int,
            default=1, help='user baseline success constant term')
    parser.add_argument('--const-pop', metavar='value', type=int,
            default=1, help='page baseline popularity constant term')
    parser.add_argument('--stop-exp', metavar='value', type=np.double,
            default=1.0, help='probability of stopping expression exponent')
    parser.add_argument('--rollback-prob', metavar='prob',
            type=np.double, default=0.5, help='roll-back probability')
    return parser

def print_arguments(args): # print useful info like how many steps to do, etc.
    for k,v in args._get_kwargs():
        print >> sys.stderr, '%s: %s' % (k.upper().replace('_',' '), str(v))
    print >> sys.stderr, 'TOTAL TIME: %9.10g days' % (args.time_step *
            args.num_steps)
    print >> sys.stderr, 'AVG BASELINE LIFETIME: %g days' % ( args.time_step *
            args.p_stop ** -1)
 
def check_arguments(args):
    if args.seed is not None and args.seed < 0:
        raise ValueError('seed cannot be negative (-s)')
    if args.num_users < 0:
        raise ValueError('num_users cannot be negative (-u)')
    if args.num_pages < 0:
        raise ValueError('num_pages cannot be negative (-p)')
    if args.time_step < 0:
        raise ValueError('time_step cannot be negative (-t)')
    if args.num_steps < 0:
        raise ValueError('num_steps cannot be negative (-n)')
    if args.p_max < 0 or args.p_max > 1:
        raise ValueError('p_max must be in [0,1] (--p-max)')
    if args.p_stop < 0 or args.p_stop > 1:
        raise ValueError('p_stop must be in [0,1] (--p-stop)')
    if args.const_succ < 0:
        raise ValueError('const_succ cannot be negative (--const_succ)')
    if args.user_input_rate < 0:
        raise ValueError('user_input_rate cannot be negative (-r/--user-input-rate)')
    if args.page_input_rate < 0:
        raise ValueError('page_input_rate cannot be negative (-R/--page-input-rate)')
    if args.confidence < 0 or args.confidence > 1:
        raise ValueError('confidence must be in [0,1] (-c/--confidence)')
    if args.rollback_prob < 0 or args.rollback_prob > 1:
        raise ValueError('rollback_prob must be in [0,1] (--rollback-prob)')
    if args.speed < 0 or args.speed > 0.5:
        raise ValueError('speed must be in [0, 0.5] (--speed)')
# warnings at the end
    if args.seed is None:
        warn('no seed specified!', category=UserWarning)
    if args.p_max == 0:
        warn('null p_max inserted: no user interactions!', category=UserWarning)
    if args.p_max == 1:
        warn('full p_max inserted: simulation may take *very* long time!', 
                category=UserWarning)
    if args.p_stop == 0:
        warn('users will not stop: simulation may take *very* long time!', 
                category=UserWarning)
    if args.p_stop == 1:
        warn('users will stop immediately', category=UserWarning)
    if args.user_input_rate == 0:
        warn('no user input', category=UserWarning)
    if args.page_input_rate == 0:
        warn('no page input', category=UserWarning)
    if args.confidence == 0:
        warn('interactions always result in failure', category=UserWarning)
    if args.confidence == 1:
        warn('interactions always result in success', category=UserWarning)
    if args.rollback_prob == 0:
        warn('no rollback interactions', category=UserWarning)
    if args.rollback_prob == 1:
        warn('always rollback interactions', category=UserWarning)
    if args.speed == 0:
        warn('null opinion update', category=UserWarning)

__all__ = [
        'make_parser',
        'check_arguments',
        'print_arguments',
]

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        check_arguments(ns)
        print_arguments(ns)
        if not ns.dry_run:
            if ns.profile:
                import pstats, cProfile
                fn = ns.profile_file or __file__ + ".prof"
                cProfile.runctx('simulate(ns)', globals(), locals(), fn)
                stats = pstats.Stats(fn)
                stats.strip_dirs().sort_stats("time").print_stats()
            else:
                prng, users, pages = simulate(ns)
    except:
        ty, val, tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            name = ty.__name__
            print >> sys.stderr, '\n%s: %s\n' % (name, val)
    finally:
        if ns.info_file:
            ns.info_file.close()
