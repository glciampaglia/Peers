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

cdef int USER_ID_MAX = 0

cdef class User:
    cdef public double opinion, edits, successes, p_activ
    cdef public int id
    def __init__(self, 
            object args, 
            object prng, 
            double edits, 
            double successes,
            object opinion, 
            object p_activ):
        global USER_ID_MAX
        self.edits = edits
        self.successes = successes
        if opinion is None:
            self.opinion = prng.rand()
        else:
            self.opinion = <double> opinion
        if p_activ is None:
            self.p_activ = prng.rand() * args.p_max
        else:
            self.p_activ = <double> p_activ
        self.id = USER_ID_MAX
        USER_ID_MAX += 1


@cython.profile(False)
cdef inline double ratio(User user) except? -1:
    cdef double den = user.edits
    cdef double num = user.successes
    assert num <= den, "user will never stop"
    if den:
        return num / den
    else:
        return 1.

cdef int PAGE_ID_MAX = 0

cdef class Page:
    cdef public double opinion, edits
    cdef public int id
    def __init__(self, object args, object prng, double edits, double opinion):
        global PAGE_ID_MAX
        self.opinion = opinion
        self.edits = edits
        self.id = PAGE_ID_MAX
        PAGE_ID_MAX += 1

cdef interaction(object args, object prng, object users, object pages, object pairs, int update_opinions):
    cdef User u
    cdef Page p
    cdef double speed = args.speed
    cdef double rollback_prob = args.rollback_prob
    cdef double confidence = args.confidence
    for i, j in pairs:
        u = users[i]
        p = pages[j]
        u.edits += 1
        p.edits += 1
        if update_opinions:
            if p.opinion < 0: # first edit of page
                p.opinion = u.opinion
                u.successes += 1
            else: # subsequent edits of page
                ok = abs(u.opinion - p.opinion) < confidence
                if ok:
                    u.successes += 1
                    u.opinion += speed * ( p.opinion - u.opinion )
                    p.opinion += speed * ( u.opinion - p.opinion )
                elif prng.rand() < rollback_prob:
                    p.opinion += speed * ( u.opinion - p.opinion )
            print args.time, u.id, p.id
        args.noedits += 1
        users[i] = u
        pages[j] = p

@cython.boundscheck(False)
@cython.wraparound(False)
cdef object selection(object args, object prng, object users, object pages):
    '''
    Returns a sequence of pairs (user, page) for interactions. 
    '''
    if len(pages) == 0:
        return []
    cdef cnp.ndarray[cnp.double_t] rvs = prng.rand(len(users))
    cdef object editing_users = deque()
    cdef int i
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

cdef update(object args, object prng, object users, object pages):
    # NOTE < τ > = Δt * p-stop**-1 ==> p-stop = Δt / < τ >
    cdef cnp.ndarray[cnp.double_t] rvs = prng.rand(len(users))
    cdef int removed = 0, idx, i
    cdef User u
    cdef double stop_exp = args.stop_exp, p_stop = args.p_stop
    for i in xrange(len(users)):
        idx = i - removed # deleting shrinks the list so need to update idx
        u = <User> users[idx]
        if rvs[i] <= 1 + ratio(u) * ( p_stop - 1): #** stop_exp:
            del users[idx]
            removed += 1
    users.extend([ User(args, prng, args.const_succ, args.const_succ, None,
            args.p_max) for i in xrange(prng.poisson(args.user_input_rate)) ])
    pages.extend([ Page(args, prng, args.const_pop, -1) for i in
            xrange(prng.poisson(args.page_input_rate)) ])

cdef step_forward(args, prng, users, pages, transient):
    cdef double dt = args.time_step
    cdef int steps, step
    if transient:
        steps = args.num_transient_steps 
    else:
        steps = args.num_steps
    for step in xrange(steps):
        pairs = selection(args, prng, users, pages)
        interaction(args, prng, users, pages, pairs, 1-transient)
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

cpdef simulate(args):
    prng = np.random.RandomState(args.seed)
    users = [ User(args, prng, args.const_succ, args.const_succ, prng.rand(),
            args.p_max) for i in xrange(args.num_users) ]
    pages = [ Page(args, prng, args.const_pop, -1) for i in 
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
