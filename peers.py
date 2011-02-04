#!/usr/bin/env python
# coding=utf-8

# file: peers.py
# vim:ts=8:sw=4:sts=4

# Come sono gestiti gli edit conflicts in Wikipedia? Vedere qui:
# http://en.wikipedia.org/wiki/Help:Edit_conflict 

''' Pure Python version '''

from __future__ import division
from argparse import ArgumentParser, FileType, SUPPRESS
import numpy as np
import sys
from time import time
from cStringIO import StringIO

from rand import randwpmf
from timeutils import si_str
from myio import arrayfile
from utils import ttysize

def myformatwarning(*args):
    msg = args[0]
    return '* WARNING: %s\n' % msg.args[0]

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
            'rate',         # edit rate
            'id',           # User id
    ]
    ID = 0                  # instance ID
    def __new__(cls, edits, successes, opinion, rate):
        self = object.__new__(cls)
        self.id = cls.ID
        cls.ID += 1
        return self
    def __init__(self, edits, successes, opinion, rate):
        self.edits = edits
        self.successes = successes
        self.opinion = opinion 
        self.rate = rate
    @property
    def ratio(self):
        den = self.edits
        num = self.successes
        if num > den:
            raise ValueError("not a probability")
        if den > 0:
            return num / den
        else:
            return 1.

class Page(object):
    __slots__ = [
            'opinion',  # see User
            'edits',    # see User
            'id',       # see User
    ]
    ID = 0              # see User
    def __new__(cls, edits, opinion):
        self = object.__new__(cls)
        self.id = cls.ID
        cls.ID += 1
        return self
    def __init__(self, edits, opinion):
        self.opinion = opinion
        self.edits = edits

def update(args, users, pages, prng=np.random):
    ''' removes active users + adds new users and pages '''
    # NOTE < τ > = Δt * p-stop**-1 ==> p-stop = Δt / < τ >
    rvs = prng.rand(len(users))
    removed = 0
    for i in xrange(len(users)):
        # deleting shrink the list so need to update idx
        idx = i - removed 
        if rvs[i] <= 1 + users[idx].ratio * (args.p_stop - 1):
            del users[idx]
            removed += 1
    # new users are added with uniformly distributed opinions
    new_users = prng.poisson(args.user_input_rate * args.time_step)
    user_opinions = prng.random_sample(new_users)
    users.extend([ User(args.const_succ, args.const_succ, o, args.edit_rate) 
            for o in user_opinions ])
    # new pages are added and for each page a user is drawn randomly with
    # replacement as the creator
    if len(users) > 0:
        new_pages = prng.poisson(args.user_input_rate * args.time_step)
        editing_users = prng.randint(0,len(users),new_pages)
        pages.extend([ Page(args.const_pop, users[i].opinion) 
                for i in editing_users ])

def times(l, T, prng=np.random):
    '''
    Sample times of increments in interval [0, T)

    Parameters
    ----------
    l - activity rate
    T - interval duration
    '''
    if l < 0:
        raise ValueError('negative rate')
    if T < 0:
        raise ValueError('negative duration')
    if l == 0:
        return []
    # N is the expected number of edits in the interval [0, T) rounded up to the
    # to the next integer, in order to avoid infinite loop if N < .5.
    N = np.ceil(l * T)
    # Draw an array of inter-times of size 2 * N
    s = (1 - np.log(prng.random_sample(2 * N))) / (2 * l)
    while np.sum(s) < T:
        # In the unlikely case that the drawn chunck is not big enough to reach
        # the end of the interval, do it again ...
        s1 = (1 - np.log(prng.random_sample(2 * N))) / (2 * l)
        s = np.hstack([s, s1])
    # Sum up to get the times
    t = np.cumsum(s)
    return t[t < T]

def pairs(n, users, pages, prng=np.random):
    '''
    Samples n indipendent (user, page) pairs with replacement

    Parameters
    ----------
    n       - sample size
    users   - either an array pmf, or an integer N. If an integer, draw n times
              from N users else, draw n times from pmf 
    pages   - ditto
    '''
    if not np.isscalar(n):
        raise ValueError('expecting a scalar n, not %s' % type(n))
    if np.isscalar(users):
        u = prng.randint(0,users,n)
    else:
        u = randwpmf(users, n, prng)
    if np.isscalar(pages):
        p = prng.randint(0,pages,n)
    else:
        p = randwpmf(pages, n, prng)
    return zip(u,p)

def edits(times, pairs, users, pages, args, nosucc=False, prng=np.random):
    '''
    Update the state of interacting users and pages

    Parameters
    ----------
    times  - editing times
    pairs  - editing pairs
    users  - list of User instances
    pages  - list of Page instances
    args   - model parameters
    nosucc - if True, edits and successes are not updated
    '''
    if nosucc:
        # turn off the update of 'edits' and 'successes', do not print.
        for t, (i, j) in zip(times, pairs):
            u = users[i]
            p = pages[j]
            if np.abs(u.opinion - p.opinion) < args.confidence:
                u.opinion += args.speed * ( p.opinion - u.opinion )
                p.opinion += args.speed * ( u.opinion - p.opinion )
            elif prng.rand() < args.rollback_prob:
                p.opinion += args.speed * ( u.opinion - p.opinion )
            users[i] = u
            pages[j] = p
    else:
        # simulation
        for t, (i, j) in zip(times, pairs):
            u = users[i]
            p = pages[j]
            u.edits += 1
            p.edits += 1
            if np.abs(u.opinion - p.opinion) < args.confidence:
                u.successes += 1
                u.opinion += args.speed * ( p.opinion - u.opinion )
                p.opinion += args.speed * ( u.opinion - p.opinion )
            elif prng.rand() < args.rollback_prob:
                p.opinion += args.speed * ( u.opinion - p.opinion )
            users[i] = u
            pages[j] = p
            print t, u.id, p.id

def loop(steps, args, users, pages, nosucc=False, prng=np.random):
    ''' 
    Main simulation loop 
    
    Parameters
    ----------
    steps   - number of update steps
    args    - model parameters
    users   - list of User instances
    pages   - list of Page instances
    noedits - if True, do not call edits()
    '''
    for i in xrange(steps):
        l = np.sum([u.rate for u in users])
        page_edits = [p.edits for p in pages]
        e = np.sum(page_edits)
        if l > 0 and e > 0:
            t = times(l, args.time_step, prng) + args.time
            if len(t) > 0:
                p = pairs(len(t), len(users), page_edits, prng)
                edits(t, p, users, pages, args, nosucc, prng)
        update(args, users, pages, prng)
        if args.info_file and args.info_binary:
            args.info_array[args.tot_steps + i] = (args.time, len(users),
                    len(pages))
        elif args.info_file:
            info = {
                    'time'  : args.time,
                    'users' : len(users),
                    'pages' : len(pages),
            }
            args.info_file.write('%(time)s %(users)s %(pages)s\n' % info)
        args.time += args.time_step
    args.tot_steps += steps

def simulate(args):
    '''
    Sets up initial population of users/pages, perform transient loop,
    then actual simulation loop. 
    '''
    prng = np.random.RandomState(args.seed)
    # users have a fixed activity rate and an initial number of ``successes''
    users = [ User(args.const_succ, args.const_succ, o, args.edit_rate) 
            for o in prng.random_sample(args.num_users) ]
    # pages have an initial value of popularity.
    pages = [ Page(args.const_pop,o) for o in prng.random_sample(args.num_pages) ]
    args.time = 0.0
    args.tot_steps = 0
    # transient
    if args.num_transient_steps > 0:
        transient_start_time = time()
        loop(args.num_transient_steps, args, users, pages, nosucc=True, 
                prng=prng)
        transient_end_time = time()
        transient_duration = transient_end_time - transient_start_time
        transient_steps_speed = args.num_transient_steps / transient_duration 
        if args.verbosity > 0:
            print >> sys.stderr, 'Transient done in %.2g sec, '\
                    '%g steps/sec' % (transient_duration, transient_steps_speed )
    else:
        if args.verbosity > 0:
            print >> sys.stderr, 'Skipping transient.'
    # simulation
    simulation_start_time = time()
    loop(args.num_steps, args, users, pages, prng=prng)
    simulation_end_time = time()
    simulation_duration = simulation_end_time - simulation_start_time
    simulation_steps_speed = args.num_steps / simulation_duration
    sys.stdout.flush()
    if args.verbosity > 0:
        print >> sys.stderr, 'Simulation done in %.2g sec, '\
                '%g steps/sec' % (simulation_duration, simulation_steps_speed)
    return prng, users, pages

class Arguments(object):
    '''
    Class for checking parameter values, and printing results of the simulation. 
    '''
    def __new__(cls, args):
        self = object.__new__(cls)
        self.__dict__.update(args.__dict__)
        return self
    def __init__(self, args):
        # __new__ takes care to update the instance __dict__
        self._check()
        self.num_steps = int(np.ceil(self.time / self.time_step))
        assert self.num_steps >= 0, "negative number of simulation steps"
        self.num_transient_steps = int(np.ceil(self.transient / self.time_step))
        assert self.num_transient_steps >= 0,\
                "negative number of transient steps"
        self.p_stop = self.time_step / self.base_life
        assert self.p_stop >= 0 and self.p_stop <= 1, "not a probability"
        self.edit_rate = self.daily_edits * self.time_step 
        assert self.edit_rate >= 0, "not a rate"
        self.user_input_rate = self.daily_users * self.time_step
        assert self.user_input_rate >= 0, "not a rate"
        self.page_input_rate = self.daily_pages * self.time_step
        assert self.page_input_rate >= 0, "not a rate"
        if self.info_binary:
            shape = (self.num_steps + self.num_transient_steps,)
            self.info_array = arrayfile(self.info_file, shape, 
                    self.info_dty_descr)
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
        if self.time_step <= 0:
            raise ValueError('update step duration cannot be negative '\
                    '(-t/--time-step)')
        if self.time < self.time_step:
            raise ValueError('simulation duration cannot be shorter than time '\
                    'step (-t/--time-step)')
        if self.base_life < self.time_step:
            raise ValueError('base life (-b/--base-life) time cannot be '\
                    'shorter than time step (-t/--time-step)')
        if self.daily_edits < 0:
            raise ValueError('daily edits cannot be negative (-e/--daily-edits)')
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
            raise ValueError('user_input_rate cannot be negative '\
                    '(-U/--daily-users)')
        if self.daily_pages < 0:
            raise ValueError('page_input_rate cannot be negative '\
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
        h,w = ttysize() or (50,85)
        sio = StringIO()
        transient_time = self.num_transient_steps * self.time_step
        print >> sio, '-'*w
        print >> sio, 'TIME.\tSimulation: %g (days).\tTransient: %g (days).\t'\
                'Update steps: %g (days)'\
                % (self.time, transient_time, self.time_step)
        print >> sio, 'USERS.\tInitial: %d (users).\tIn-rate: %g (users/day).\t'\
                'Activity: %g (edits/day)' % (self.num_users, self.daily_users,
                        self.daily_edits)
        print >> sio, '\tBase life: %g (days)' % self.base_life
        print >> sio, 'PAGES.\tInitial: %d (pages).\tIn-rate: %g (pages/day).' % (
                self.num_pages, self.daily_pages)
        print >> sio, 'PAIRS.\tBase success: %g.\tBase popularity: %g.'\
                % (self.const_succ, self.const_pop)
        print >> sio, 'EDITS.\tConfidence: %g.\tSpeed: %g.\t\tRollback-prob.: %g.'\
                % (self.confidence, self.speed, self.rollback_prob)
        print >> sio, 'MISC.\tSeed: %s\t\tInfo file: %s.\tBinary mode: %s' % (self.seed,
                self.info_file.name if self.info_file else 'None', 'ON' if self.info_binary else 'OFF')
        try:
            import sys
            sys.stderr = sio
            self._warn()
        finally:
            sys.stderr = sys.__stderr__
        print >> sio, '-'*w,
        return sio.getvalue()

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
            help='simulation duration, in days')
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
            metavar='DAYS',
            help='duration of population update interval' )
    parser.add_argument(
            '-e',
            '--daily-edits',
            type=float,
            metavar='EDITS',
            help='average daily number of edits of a user',
            default=1)
    parser.add_argument(
            '-b',
            '--base-life',
            type=float, 
            metavar='DAYS',
            help='asymptotic average user lifetime',
            default=100.0)
    parser.add_argument(
            '-u',
            '--num-users',
            type=int,
            default=0,
            help='initial number of users',
            metavar='USERS')
    parser.add_argument(
            '-p',
            '--num-pages',
            type=int,
            default=0,
            help='initial number of pages',
            metavar='PAGES')
    parser.add_argument(
            '-U',
            '--daily-users',
            metavar='USERS',
            default=1.0, 
            type=np.double,
            help='daily rate of new user arrivals')
    parser.add_argument(
            '-P',
            '--daily-pages',
            metavar='PAGES',
            type=np.double,
            default=1.0,
            help='daily rate of new page creations')
    parser.add_argument(
            '-c',
            '--confidence',
            metavar='VALUE',
            type=np.double,
            default=.2,
            help='confidence parameter')
    parser.add_argument(
            '-s',
            '--speed',
            metavar='VALUE',
            type=np.double,
            default=0.5,
            help='opinion averaging speed')
    parser.add_argument(
            '-T',
            '--transient',
            type=int,
            metavar='DAYS',
            help='transient duration',
            default=0)
    parser.add_argument(
            '--const-succ',
            metavar='EDITS',
            type=float,
            default=1,
            help='user baseline success constant term')
    parser.add_argument(
            '--const-pop',
            metavar='EDITS',
            type=float,
            default=1,
            help='page baseline popularity constant term')
    parser.add_argument(
            '--rollback-prob',
            metavar='PROB',
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
            metavar='FILE')
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
            help='do not simulate, just print parameters defaults')
    parser.add_argument(
            '-D',
            '--debug',
            action='store_true',
            help='raise Python exceptions to the console')
    parser.add_argument(
            '--profile',
            action='store_true',
            help='run profiling')
    parser.add_argument(
            '--profile-file',
            metavar='FILE',
            default=None,
            help="store profiling information in file")
    parser.add_argument(
            '-n',
            '--no-banner',
            action='store_const',
            const=1,
            dest='verbosity',
            help='do not print banner.')
    parser.add_argument(
            '-q',
            '--quiet',
            action='store_const',
            const=0,
            dest='verbosity',
            help='do not print the banner')
    parser.set_defaults(verbosity=2)
    return parser

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        ns = Arguments(ns) # will check argument values here
        if ns.verbosity > 1:
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
