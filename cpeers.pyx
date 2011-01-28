# coding=utf-8
# cython: profile=True

# file: cpeers.pyx
# vim:ts=8:sw=4:sts=4

''' Cython implementation '''

from __future__ import division
import numpy as np
from collections import deque
import sys
import cython
from time import time
from rand import randwpmf

# cimports
# from rand cimport *
cimport numpy as cnp

cdef extern from "math.h":
    double log(double)

cdef int USER_ID_MAX = 0

cdef class User:
    cdef public double opinion, edits, successes, rate
    cdef public int id
    def __init__(self, 
            double edits, 
            double successes,
            object opinion, 
            object rate):
        global USER_ID_MAX
        self.edits = edits
        self.successes = successes
        self.opinion = opinion
        self.rate = rate
        self.id = USER_ID_MAX
        USER_ID_MAX += 1

@cython.profile(False)
cdef inline double ratio(User user) except? -1:
    cdef double den = user.edits
    cdef double num = user.successes
    if num > den:
        raise ValueError("not a probability")
    if den:
        return num / den
    else:
        return 1.

cdef int PAGE_ID_MAX = 0

cdef class Page:
    cdef public double opinion, edits
    cdef public int id
    def __init__(self, double edits, double opinion):
        global PAGE_ID_MAX
        self.opinion = opinion
        self.edits = edits
        self.id = PAGE_ID_MAX
        PAGE_ID_MAX += 1

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef int update(object args, object users, object pages, object prng) except 1:
    cdef cnp.ndarray[cnp.double_t] rvs = prng.rand(len(users))
    cdef cnp.ndarray[cnp.double_t] user_opinions
    cdef cnp.ndarray[cnp.int_t] editing_users
    cdef int removed = 0, idx, i, new_users, new_pages
    cdef User u
    cdef p_stop = args.p_stop
    for i in xrange(len(users)):
        # deleting shrinks the list so need to update idx
        idx = i - removed 
        u = <User> users[idx]
        if rvs[i] <= 1 + ratio(u) * ( p_stop - 1): 
            del users[idx]
            removed += 1
    new_users = prng.poisson(args.user_input_rate * args.time_step)
    user_opinions = prng.random_sample(new_users)
    users.extend([ User(args.const_succ, args.const_succ, o, args.edit_rate)
            for o in user_opinions ])
    if len(users) > 0:
        new_pages = prng.poisson(args.page_input_rate * args.time_step)
        editing_users = prng.randint(0, len(users), new_pages)
        pages.extend([ Page(args.const_pop, users[i].opinion) for i in
            editing_users])
    return 0

cdef inline object _intertimes(double l, object u):
    cdef cnp.ndarray[cnp.double_t] _u = u
    cdef int n = len(_u), i
    for i in xrange(n):
        _u[i] = (1.0 - log(_u[i])) / (2.0 * l)
    return _u

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef object times(double l, double T, object prng):
    cdef double N
    cdef cnp.ndarray[cnp.double_t] s, s1
    if l < 0:
        raise ValueError('negative rate')
    if T < 0:
        raise ValueError('negative duration')
    if l == 0:
        return []
    N = np.ceil(l * T)
    s = _intertimes(l, prng.random_sample(2 * N))
    while np.sum(s) < T:
        s1 = _intertimes(l, prng.random_sample(2 * N))
        s = np.hstack([s, s1])
    t = np.cumsum(s)
    return t[t < T]

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef object pairs(int n, object users, object pages, object prng):
    cdef cnp.ndarray[cnp.int_t] u,p
    if not np.isscalar(n):
        raise ValueError('expecting a scalar n, not %s' % type(n))
    if np.isscalar(users):
        u = prng.randint(0, users, n)
    else:
        u = randwpmf(users, n, prng)
    if np.isscalar(pages):
        p = prng.randint(0, pages, n)
    else:
        p = randwpmf(pages, n, prng)
    return zip(u, p)

#@cython.boundscheck(False)
#@cython.wraparound(False)
cdef int edits(
        object times, 
        object pairs, 
        object users, 
        object pages, 
        object args, 
        int nosucc, 
        object prng) except 1:
    cdef User u
    cdef Page p
    cdef double speed = args.speed, rollback_prob = args.rollback_prob,\
            confidence = args.confidence
    cdef int k
    if nosucc:
        # turn off the update of 'edits' and 'successes', do not print.
        for k in xrange(len(times)):
            t = times[k]
            i, j = pairs[k]
            u = users[i]
            p = pages[j]
            if abs(u.opinion - p.opinion) < confidence:
                u.opinion += speed * ( p.opinion - u.opinion )
                p.opinion += speed * ( u.opinion - p.opinion )
            elif prng.rand() < rollback_prob:
                p.opinion += speed * ( u.opinion - p.opinion )
            users[i] = u
            pages[j] = p
    else:
        # simulation
        for k in xrange(len(times)):
            t = times[k]
            i, j = pairs[k]
            u = users[i]
            p = pages[j]
            u.edits += 1
            p.edits += 1
            if abs(u.opinion - p.opinion) < confidence:
                u.successes += 1
                u.opinion += speed * ( p.opinion - u.opinion )
                p.opinion += speed * ( u.opinion - p.opinion )
            elif prng.rand() < rollback_prob:
                p.opinion += speed * ( u.opinion - p.opinion )
            users[i] = u
            pages[j] = p
            print t, u.id, p.id
    return 0

cdef int loop(
        int steps, 
        object args, 
        object users, 
        object pages, 
        int nosucc, 
        object prng) except -1:
    cdef double l
    cdef int n,m,k,i,e
    cdef User u
    cdef cnp.ndarray[cnp.int_t] page_edits
    for i in xrange(steps):
        n = len(users)
        l = 0.0
        for k in xrange(n):
            u = users[k]
            l += u.rate
        m = len(pages)
        page_edits = np.empty(m, dtype=int)
        e = 0
        for k in xrange(m):
            page_edits[k] = pages[k].edits
            e += page_edits[k]
        if l > 0.0 and e > 0:
            t = times(l, args.time_step, prng) + args.time
            if len(t) > 0:
                p = pairs(len(t), n, page_edits, prng)
                edits(t, p, users, pages, args, nosucc, prng)
        update(args, users, pages, prng)
        if args.info_file and args.info_binary:
            args.info_array[args.elapsed_steps] = (args.time, len(users),
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
    return i+1

cpdef simulate(args):
    cdef int steps = 0
    prng = np.random.RandomState(args.seed)
    users = [ User(args.const_succ, args.const_succ, prng.rand(), args.edit_rate)
            for i in xrange(args.num_users) ]
    pages = [ Page(args.const_pop, prng.rand()) for i in xrange(args.num_pages) ] 
    args.time = 0.0
    args.tot_steps = 0
    # transient
    if args.num_transient_steps > 0:
        transient_start_time = time()
        loop(args.num_transient_steps, args, users, pages, 1, prng)
        transient_end_time = time()
        transient_duration = transient_end_time - transient_start_time
        transient_steps_speed = args.num_transient_steps / transient_duration
        if args.verbosity > 0:
            print >> sys.stderr, 'Transient done in %.2g sec, '\
                    '%g steps/sec' % (transient_duration, transient_steps_speed) 
    else:
        if args.verbosity > 0:
            print >> sys.stderr, "Skipping transient."
    # simulation
    simulation_start_time = time()
    loop(args.num_steps, args, users, pages, 0, prng)
    simulation_end_time = time()
    simulation_duration = simulation_end_time - simulation_start_time
    simulation_steps_speed = args.num_steps / simulation_duration
    sys.stdout.flush()
    if args.verbosity > 0:
        print >> sys.stderr, 'Simulation done in %.2g sec, '\
                '%g steps/sec' % (simulation_duration, simulation_steps_speed)
    return prng, users, pages

