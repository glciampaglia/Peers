#!/usr/bin/env python
# coding=utf-8

# file: peers.py
# vim:ts=8:sw=4:sts=4

# TODO <Fri Dec  3 12:42:10 CET 2010>
# 1. Arguments should raise ArgumentError in the _check method. ArgumentError
#    instances take as argument the name of the variable being tested. This
#    information will be useful for reporting the actual ArgumentParser action
#    related to this variable.
# 2. The try/except block in the __main__ clause should check if the raised
#    exception is instance of ArgumentError. If that is the case, should lookup
#    the variable name within the actions of the parser, and generate an error
#    message that contains the switches (for optionals) or the metavar (for
#    positionals) that the user should check.

''' Pure Python version '''

from __future__ import division
from argparse import ArgumentParser, FileType, SUPPRESS
import numpy as np
from collections import deque
import sys
from time import time
from cStringIO import StringIO

from rand import randwpmf
from timeutils import si_str
from myio import arrayfile

def myformatwarning(*args):
    msg = args[0]
    return '*** WARNING *** %s\n' % msg.args[0]

# custom warning formatting
formatwarning = myformatwarning
import warnings
# default is saved in warnings._formatwarning
warnings._formatwarning = warnings.formatwarning
warnings.formatwarning = formatwarning
from warnings import warn

__all__ = [
        'make_parser',
        'Arguments',
]

class User(object):
    __slots__ = [           # this saves memory
            'opinion',      # \in [0,1], named this way for historic reasons
            'edits',        # \ge 0, number of edits performed
            'successes',    # \ge 0 and \le edits, number of successful edits
            'p_activ',      # \le p_max, probability of activation in dt
            'id',           # User id
    ]
    USER_ID_MAX = 0
    def __init__(self, args, prng, edits, successes, opinion=None, p_activ=None):
        self.edits = edits
        self.successes = successes
        self.opinion = opinion or prng.rand()               # ~ U[0,1]
        self.p_activ = p_activ or prng.rand() * args.p_max  # ~ U[0,p_max]
        self.id = User.USER_ID_MAX
        User.USER_ID_MAX += 1
    @property
    def ratio(self):
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
            'id',       # see User
    ]
    PAGE_ID_MAX = 0
    def __init__(self, args, prng, edits, opinion=None):
        self.opinion = opinion
        self.edits = edits
        self.id = Page.PAGE_ID_MAX
        Page.PAGE_ID_MAX += 1

class Arguments(object):
    '''
    Class for condition checking and printing
    '''
    def __new__(cls, args):
        self = object.__new__(cls)
        self.__dict__.update(args.__dict__)
        return self
    def __init__(self, args):
        # __new__ takes care to update the instance __dict__
        self._check()
        self.num_steps = int(np.ceil(self.time / self.time_step))
        assert self.num_steps >= 0, "not positive"
        self.p_stop = self.time_step / self.base_life
        assert self.p_stop >= 0 and self.p_stop <= 1, "not a probability"
        self.p_max = self.daily_edits * self.time_step 
        assert self.p_max >= 0 and self.p_max <= 1, "not a probability"
        self.user_input_rate = self.daily_users * self.time_step
        assert self.user_input_rate >= 0, "not a rate"
        self.page_input_rate = self.daily_pages * self.time_step
        assert self.page_input_rate >= 0, "not a rate"
        if args.info_binary:
            shape = (args.num_steps + args.num_transient_steps,)
            args.info_array = arrayfile(args.info_file, shape, args.info_dty_descr)
    def _check(self):
        #--------#
        # Errors #
        #--------#
        if self.seed is not None and self.seed < 0:
            raise ValueError('seed cannot be negative (-s)')
        if self.time_step <= 0:
            raise ValueError('time step cannot be negative (-t/--time-step)')
        if self.time < self.time_step:
            raise ValueError('simulation time cannot be shorter than time step (-t/--time-step)')
        if self.base_life < self.time_step:
            raise ValueError('base life (-b/--base-life) time cannot be shorter than time step (-t/--time-step)')
        if self.daily_edits < 0:
            raise ValueError('daily edits cannot be negative (-e/--daily-edits)')
        if self.daily_edits * self.time_step > 1:
            raise ValueError('time step is too short to simulate this daily rate of edits (-e/--daily-edits)')
        if self.num_users < 0:
            raise ValueError('num_users cannot be negative (-u/--num-users)')
        if self.num_pages < 0:
            raise ValueError('num_pages cannot be negative (-p/--num-pages)')
        if self.const_succ < 0:
            raise ValueError('const_succ cannot be negative (--const_succ)')
        if self.daily_users < 0:
            raise ValueError('user_input_rate cannot be negative (-U/--daily-users)')
        if self.daily_pages < 0:
            raise ValueError('page_input_rate cannot be negative (-P/--daily-pages)')
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
        sio = StringIO()
        for k,v in sorted(self.__dict__.iteritems()):
            if k in [ 'time_step', 'time', 'base_life' ]:
                v = si_str(v)
            print >> sio, '%s: %s' % (k.upper().replace('_',' '), v)
        try:
            import sys
            sys.stderr = sio
            self._warn()
        finally:
            sys.stderr = sys.__stderr__
        return sio.getvalue()

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
            print args.time, u.id, p.id
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
    editing_pages = randwpmf(page_pmf, num=len(editing_users), prng=prng)
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
        if rvs[i] <= 1 + users[idx].ratio * (args.p_stop - 1):
            del users[idx]
            removed += 1
    # create users with fixed activity rate
    users.extend([ User(args, prng, args.const_succ, args.const_succ, None,
            args.p_max) for i in xrange(prng.poisson(args.user_input_rate)) ])
    pages.extend([ Page(args, prng, args.const_pop) for i in
            xrange(prng.poisson(args.page_input_rate)) ])

def step_forward(args, prng, users, pages, transient):
    dt = args.time_step
    if transient:
        steps = args.num_transient_steps 
    else:
        steps = args.num_steps
    for step in xrange(steps):
        pairs = selection(args, prng, users, pages)
        interaction(args, prng, users, pages, pairs, update_opinions=1-transient)
        update(args, prng, users, pages)
        if args.info_file and args.info_binary:
            args.info_array[args.elapsed_steps] = (args.time, len(users),
                    len(pages))
        elif args.info_file:
            info = {}
            info['time'] = args.time
            info['users'] = len(users)
            info['pages'] = len(pages)
            args.info_file.write('%(time)s %(users)s %(pages)s\n' % info)
        args.time += args.time_step
        args.elapsed_steps += 1

def simulate(args):
    prng = np.random.RandomState(args.seed)
    user_op = prng.rand(args.num_users)
    page_op = prng.rand(args.num_pages)
    users = [ User(args, prng, args.const_succ, args.const_succ, user_op[i],
            args.p_max) for i in xrange(args.num_users) ]
    pages = [ Page(args, prng, args.const_pop, page_op[i]) for i in 
            xrange(args.num_pages) ] 
    args.time = 0.0
    start_time = time()
    args.elapsed_steps = 0
    args.noedits = 0
    try:
        step_forward(args, prng, users, pages, 1) # don't output anything
        step_forward(args, prng, users, pages, 0) # actual simulation output
    finally:
        speed = args.elapsed_steps / ( time() - start_time )
        activity = args.noedits / (args.elapsed_steps * args.time_step )
        print >> sys.stderr, " *** Speed: %g (steps/sec)" % speed
        print >> sys.stderr, " *** Activity: %g (edits/simulated day)" % activity
    return prng, users, pages

desc = 'The `Peers\' agent-based model © (2010) G.L. Ciampaglia'
usage = '%(prog)s [OPTIONS, @file] duration [seed]'

def make_parser(): 
    parser = ArgumentParser(description=desc, usage=usage, 
            fromfile_prefix_chars='@')
    #----------------------#
    # positional arguments #
    #----------------------#
    parser.add_argument(
            'time',
            type=float, 
            help='simulated time, in number of days')
    parser.add_argument(
            'seed',
            type=int,
            nargs='?',
            help='seed of the pseudo-random numbers generator',
            metavar='seed')
    #--------------------------------------#
    # optional arguments (model parameter) #
    #--------------------------------------#
    parser.add_argument(
            '-t',
            '--time-step',
            type=np.double,
            default=1.0, 
            metavar='value',
            help='simulation update time step Δt (in days)' )
    parser.add_argument(
            '-e',
            '--daily-edits',
            type=float,
            help='average daily number of edits per user ',
            default=1)
    parser.add_argument(
            '-b',
            '--base-life',
            type=float, 
            help='baseline average user lifetime',
            default=100.0)
    parser.add_argument(
            '-u',
            '--num-users',
            type=int,
            default=0,
            help='initial number of users',
            metavar='value')
    parser.add_argument(
            '-p',
            '--num-pages',
            type=int,
            default=0,
            help='initial number of pages',
            metavar='value')
    parser.add_argument(
            '-U',
            '--daily-users',
            metavar='rate',
            default=1.0, 
            type=np.double,
            help='daily rate of new users')
    parser.add_argument(
            '-P',
            '--daily-pages',
            metavar='rate',
            type=np.double,
            default=1.0,
            help='daily rate of new pages')
    parser.add_argument(
            '-c',
            '--confidence',
            metavar='value',
            type=np.double,
            default=.2,
            help='confidence parameter')
    parser.add_argument(
            '-s',
            '--speed',
            metavar='value',
            type=np.double,
            default=0.5,
            help='opinion averaging speed')
    parser.add_argument(
            '--transient',
            dest='num_transient_steps',
            type=int,
            metavar='value',
            help='number of transient steps',
            default=0)
    parser.add_argument(
            '--const-succ',
            metavar='value',
            type=int,
            default=1,
            help='user baseline success constant term')
    parser.add_argument(
            '--const-pop',
            metavar='value',
            type=int,
            default=1,
            help='page baseline popularity constant term')
    parser.add_argument(
            '--rollback-prob',
            metavar='prob',
            type=np.double,
            default=0.5,
            help='roll-back probability')
    #-------------------------------------------#
    # optional arguments (simulator parameters) #
    #-------------------------------------------#
    parser.add_argument(
            '-i',
            '--info-file',
            type=FileType('w+'),
            help='write simulation info to file',
            metavar='file')
    parser.add_argument(
            '--info-binary',
            action='store_true',
            default=False,
            help='write binary data to info file (NumPy format)')
    parser.add_argument(
            'info_dty_descr',
            default='f8, i4, i4,',
            help=SUPPRESS,
            nargs='?')
    parser.add_argument(
            '-d',
            '--dry-run',
            action='store_true',
            default=False,
            help='do not simulate, just print parameters defaults')
    parser.add_argument(
            '-D',
            '--debug',
            action='store_true',
            default=False, 
            help='raise Python exceptions to the console')
    parser.add_argument(
            '--profile',
            action='store_true',
            default=False,
            help='run profiling')
    parser.add_argument(
            '--profile-file',
            metavar='file',
            default=None,
            help="store profiling information in file")
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        ns = Arguments(ns) # will check argument values here
        print >> sys.stderr, ns
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
