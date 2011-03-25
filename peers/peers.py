#!/usr/bin/env python
# coding=utf-8

# file: peers.py
# vim:ts=8:sw=4:sts=4

"""The Peers agent-based model - © 2010-2011 of Giovanni Luca Ciampaglia."""

from __future__ import division
from argparse import ArgumentParser, FileType
import numpy as np
import sys
from time import time
from cStringIO import StringIO
from warnings import warn

from .rand import randwpmf
from .utils import ttysize, IncIDMixin
from .cpeers import loop as c_loop

class User(IncIDMixin):
    ''' Class for user instances '''
    __slots__ = [
            'opinion',      # \in [0,1], named this way for historic reasons
            'edits',        # \ge 0, number of edits performed
            'successes',    # \ge 0 and \le edits, number of successful edits
            'rate',         # edit rate
    ]
    def __init__(self, edits, successes, opinion, rate):
        self.edits = edits
        self.successes = successes
        self.opinion = opinion 
        self.rate = rate
    @property
    def ratio(self):
        return self.successes / self.edits

class Page(IncIDMixin):
    ''' Class for page instances '''
    __slots__ = [
            'opinion',  # see User
            'edits',    # see User
    ]
    def __init__(self, edits, opinion):
        self.opinion = opinion
        self.edits = edits

def loop(tstart, tstop, args, users, pages, output, prng=np.random):
    ''' continuous time simulation loop '''
    t = tstart # current time
    uR = args.daily_users
    pR = args.daily_pages
    p1 = args.p_stop_long
    p2 = args.p_stop_short
    num_events = 0
    while True:
        rates = np.asarray([ (u.rate, u.ratio) for u in users ])
        if len(rates):
            rates[:, 1] = rates[:,1] * p1 + (1 - rates[:,1]) * p2
            eR, dR = np.sum(rates, axis=0)
        else:
            eR, dR = 0.0, 0.0
        R = eR + dR + uR + pR
        T = (1 - np.log(prng.uniform())) / R # time to next event
        if t + T > tstop:
            break
        t = t + T
        num_events += 1
        ev = randwpmf([eR, dR, uR, pR], prng=prng)
        if ev == 0: # edit
            if len(pages):
                user_idx = randwpmf(rates[:,0], prng=prng)
                page_idx = randwpmf([p.edits for p in pages], prng=prng)
                user = users[user_idx]
                page = pages[page_idx]
                user.edits += 1
                page.edits += 1
                if np.abs(user.opinion - page.opinion) < args.confidence:
                    user.successes += 1
                    user.opinion += args.speed * ( page.opinion - user.opinion )
                    page.opinion += args.speed * ( user.opinion - page.opinion )
                elif prng.rand() < args.rollback_prob:
                    page.opinion += args.speed * ( user.opinion - page.opinion )
                users[user_idx] = user
                pages[page_idx] = page
                if output:
                    print t, user.id, page.id
        elif ev == 1: # user stops
            user_idx = randwpmf(rates[:,1], prng=prng)
            del users[user_idx]
        elif ev == 2: # new user
            o = prng.uniform()
            user = User(args.const_succ, args.const_succ, o, args.daily_edits)
            users.append(user)
        else: # new page
            if len(users):
                user_idx = prng.randint(0, len(users))
                user = users[user_idx]
                page = Page(args.const_pop + 1, user.opinion)
                pages.append(page)
        if args.info_file is not None:
            args.info_file.write('%g %g %g\n' % (t, len(users), len(pages)))
    return num_events

def simulate(args):
    '''
    Performs one simulation.

    Parameters
    ----------
    args - an Arguments instance

    Returns
    -------
    prng, users, pages
    '''
    prng = np.random.RandomState(args.seed)
    # users have a fixed activity rate and an initial number of ``successes''
    users = [ User(args.const_succ, args.const_succ, o, args.daily_edits) 
            for o in prng.random_sample(args.num_users) ]
    # pages have an initial value of popularity.
    pages = [ Page(args.const_pop,o) for o in prng.random_sample(args.num_pages) ]
    if args.transient:
        t_transient_start = time()
        if args.fast:
            n_transient = c_loop(0, args.transient, args, users, pages, False, prng)
        else:
            n_transient = loop(0, args.transient, args, users, pages, False, prng)
        t_transient_stop = time()
        t_transient = t_transient_stop - t_transient_start
        if args.verbosity > 0:
            print >> sys.stderr, 'Transient done in %.2gs (%g events/s)'\
                    % (t_transient, n_transient / t_transient)
    t_sim_start = time()
    if args.fast:
        n_sim = c_loop(args.transient, args.transient + args.time, args, users, 
                pages, True, prng)
    else:
        n_sim = loop(args.transient, args.transient + args.time, args, users, 
                pages, True, prng)
    t_sim_stop = time()
    t_sim = t_sim_stop - t_sim_start
    if args.verbosity > 0:
        print >> sys.stderr, 'Simulation done in %.2gs (%g events/s)'\
                % (t_sim, n_sim / t_sim)
    return prng, users, pages

class Arguments(object):
    """
    Class for checking parameter values and for printing simulation's results
    """
    def __init__(self, args):
        self.__dict__.update(args.__dict__)
        self._check()
        self.p_stop_long = self.long_life ** -1
        self.p_stop_short = self.short_life ** -1
    def _check(self):
        #--------#
        # Errors #
        #--------#
        if self.seed is not None and self.seed < 0:
            raise ValueError('seed cannot be negative (-s)')
        if self.time < 0:
            raise ValueError('simulation duration cannot be negative')
        if self.transient < 0:
            raise ValueError('transient duration cannot be negative')
        if self.num_users < 0:
            raise ValueError('initial number of users cannot be negative '\
                    '(-u/--num-users)')
        if self.num_pages < 0:
            raise ValueError('num_pages cannot be negative (-p/--num-pages)')
        if self.const_succ < 0:
            raise ValueError('const_succ cannot be negative (--const-succ)')
        if self.const_pop <= 0:
            raise ValueError('const_pop must be positive (--const-pop)')
        if self.daily_users < 0:
            raise ValueError('daily_users cannot be negative '
                    '(-U/--daily-users)')
        if self.daily_pages < 0:
            raise ValueError('daily_pages cannot be negative '
                    '(-P/--daily-pages)')
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError('confidence must be in [0,1] (-c/--confidence)')
        if self.rollback_prob < 0 or self.rollback_prob > 1:
            raise ValueError('rollback_prob must be in [0,1] (--rollback-prob)')
        if self.speed < 0 or self.speed > 0.5:
            raise ValueError('speed must be in [0, 0.5] (--speed)')
    def _warn(self):
        #----------#
        # Warnings #
        #----------#
        if self.seed is None:
            warn('no seed was specified', category=UserWarning)
        if self.daily_edits == 0:
            warn('turning off user edits', category=UserWarning)
        if self.daily_users == 0:
            warn('turning off new users arrival', category=UserWarning)
        if self.daily_pages == 0:
            warn('turning off page creation', category=UserWarning)
        if self.confidence == 0:
            warn('edits will always result in failure', category=UserWarning)
        if self.confidence == 1:
            warn('edits always result in success', category=UserWarning)
        if self.rollback_prob == 0:
            warn('no rollback edits', category=UserWarning)
        if self.rollback_prob == 1:
            warn('always do rollback edits', category=UserWarning)
        if self.speed == 0:
            warn('turning off opinion update', category=UserWarning)
    def __str__(self):
        h,w = ttysize() or (50, 85)
        sio = StringIO()
        print >> sio, '-'*w
        print >> sio, 'TIME.\tSimulation: %g (days).\tTransient: %g (days).'\
                % (self.time, self.transient)
        print >> sio, 'USERS.\tInitial: %d (users).\tIn-rate: %g (users/day).'\
                '\tActivity: %g (edits/day)' % (self.num_users, self.daily_users,
                self.daily_edits)
        print >> sio, '\tLong life: %g (days)\tShort life: %g (days)'\
                % (self.long_life, self.short_life)
        print >> sio, 'PAGES.\tInitial: %d (pages).\tIn-rate: %g (pages/day).' % (
                self.num_pages, self.daily_pages)
        print >> sio, 'PAIRS.\tBase success: %g.\tBase popularity: %g.'\
                % (self.const_succ, self.const_pop)
        print >> sio, 'EDITS.\tConfidence: %g.\tSpeed: %g.\t\tRollback-prob.: %g.'\
                % (self.confidence, self.speed, self.rollback_prob)
        print >> sio, 'MISC.\tSeed: %s\t\tInfo file: %s.' % (self.seed,
                 self.info_file.name if self.info_file else 'None')
        try:
            import sys
            sys.stderr = sio
            self._warn()
        finally:
            sys.stderr = sys.__stderr__
        print >> sio, '-'*w,
        return sio.getvalue()

def make_parser(): 
    parser = ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    parser.add_argument('time', type=float, help='simulation duration, in days')
    parser.add_argument('seed', type=int, nargs='?', help='seed of the '
            'pseudo-random numbers generator', metavar='seed')
    parser.add_argument('-T', '--transient', type=float, metavar='DAYS',
            help='transient duration (default: %(default)g)', default=0.0)
    parser.add_argument('-e', '--daily-edits', type=float, metavar='EDITS',
            help='average daily number of edits of a user', default=1.0)
    parser.add_argument('-L', '--long-life', type=float, metavar='DAYS',
            help='user long-term lifespan (default: %(default)g)', default=100.0)
    parser.add_argument('-l', '--short-life', type=float, metavar='DAYS',
            help='user short-term lifespan (default: %(default)g)', 
            default=1.0/24.0)
    parser.add_argument('-u', '--users', type=int, default=0, help='initial'
            ' number of users (default: %(default)d)', dest='num_users',
            metavar='NUM')
    parser.add_argument('-p', '--pages', type=int, default=0, help='initial'
            ' number of pages (default: %(default)d)', dest='num_pages',
            metavar='NUM')
    parser.add_argument('-U', '--daily-users', metavar='RATE', default=1.0, 
            type=np.double, help='daily rate of new users (default: '
            '%(default)g)')
    parser.add_argument('-P', '--daily-pages', metavar='RATE', default=1.0, 
            type=np.double, help='daily rate of new pages (default: '
            '%(default)g)')
    parser.add_argument('-c', '--confidence', type=np.double, default=.2,
            help='confidence parameter (default: %(default)g)')
    parser.add_argument('-s', '--speed', type=np.double, default=0.5,
            help='opinion averaging speed (default: %(default)g)')
    parser.add_argument('--const-succ', metavar='EDITS', type=float,
            default=1.0, help='base user successes (default: %(default)g)')
    parser.add_argument('--const-pop', metavar='EDITS', type=float,
            default=1.0, help='base page popularity (default: %(default)g)')
    parser.add_argument('-r', '--rollback-prob', metavar='PROB', type=np.double,
            default=0.5, help='roll-back probability (default: %(default)g)')
# misc
    parser.add_argument('-d', '--dry-run', action='store_true',
            help='do not simulate, just print parameters defaults')
    parser.add_argument('-D', '--debug', action='store_true', 
            help='raise Python exceptions to the console')
    parser.add_argument('-i', '--info', type=FileType('w'), help='write '
            'simulation information to %(metavar)s', metavar='FILE',
            dest='info_file')
    parser.add_argument('--fast', action='store_true', help='Use Cython '
            'implementation')
# profiling
    parser.add_argument('--profile', action='store_true', help='run profiling')
    parser.add_argument('--profile-file', metavar='FILE', default=None,
            help="store profiling information in file")
# verbosity
    parser.add_argument('-n', '--no-banner', action='store_const', const=1,
            dest='verbosity', help='do not print banner.')
    parser.add_argument('-q', '--quiet', action='store_const', const=0,
            dest='verbosity', help='do not print the banner')
    parser.set_defaults(verbosity=2)
    return parser

def main(args):
    '''
    Parameters
    ----------
    args - a Namespace parsed from parser generated with make_parser

    Example
    -------
    >>> from peers import make_parser, main
    >>> parser = make_parser()
    >>> ns = parser.parse('10 1'.split()) # will use defaults
    >>> main(ns)
    ...
    '''
    try:
        args = Arguments(args) # will check argument values here
        if args.verbosity > 1:
            print >> sys.stderr, args
        if not args.dry_run:
            if args.profile:
                import pstats, cProfile
                fn = args.profile_file or __file__ + ".prof"
                cProfile.runctx('simulate(args)', globals(), locals(), fn)
                stats = pstats.Stats(fn)
                stats.strip_dirs().sort_stats("time").print_stats()
            else:
                prng, users, pages = simulate(args)
    except:
        ty, val, tb = sys.exc_info()
        if args.debug:
            raise ty, val, tb
        else:
            name = ty.__name__
            print >> sys.stderr, '\n%s: %s\n' % (name, val)
    finally:
        if args.info_file:
            args.info_file.close()

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
