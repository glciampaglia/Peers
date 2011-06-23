# coding=utf-8
# cython: profile=True

# file: cpeers.pyx
# vim:ts=8:sw=4:sts=4

''' Cython implementation '''

from __future__ import division
import numpy as np
import cython

from peers.rand cimport _randwpmf
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
    return num / den

cdef User toUser(object u):
    return User(u.edits, u.successes, u.opinion, u.rate)

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

cdef Page toPage(object p):
    return Page(p.edits, p.opinion)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int loop(
        double tstart,
        double tstop, 
        object args, 
        object users, 
        object pages,
        int output,
        object prng) except? -1:
    cdef int num_events = 0, ev, user_idx, page_idx
    cdef User user
    cdef Page page
    cdef double r, eR, dR, R, T, o, ups
    cdef double t = tstart
    cdef double uR = args.daily_users
    cdef double pR = args.daily_pages
    cdef double p1 = args.p_stop_long
    cdef double p2 = args.p_stop_short
    cdef double confidence = args.confidence
    cdef double speed = args.speed
    cdef double rollback_prob = args.rollback_prob
    cdef object pedit = []
    cdef object pstop = []
    cdef object ppage = []
    eR = 0.0
    dR = 0.0
    users = [ toUser(u) for u in users ]
    pages = [ toPage(p) for p in pages ]
    for i in xrange(len(users)):
        user = users[i]
        pedit.append(user.rate)
        eR += user.rate
        r = ratio(user)
        ups = r * p1 + (1 - r) * p2
        pstop.append(ups)
        dR += ups
    for i in xrange(len(pages)):
        page = <Page>pages[i]
        ppage.append(page.edits)
    while 1:
        R = eR + dR + uR + pR
        T = (1.0 - log(prng.uniform())) / R
        if t + T > tstop:
            break
        t = t + T
        num_events += 1
        ev = _randwpmf([eR, dR, uR, pR], 1, prng)
        if ev == 0: # edit
            if len(pages):
                user_idx = _randwpmf(pedit, 1, prng)
                page_idx = _randwpmf(ppage, 1, prng)
                user = users[user_idx]
                page = <Page>pages[page_idx]
                r = ratio(user)
                dR -= (r * p1 + (1 - r) * p2)
                user.edits += 1
                page.edits += 1
                if abs(user.opinion - page.opinion) < confidence:
                    user.successes += 1
                    user.opinion += speed * (page.opinion - user.opinion)
                    page.opinion += speed * (user.opinion - page.opinion)
                elif prng.rand() < rollback_prob:
                    page.opinion += speed * (user.opinion - page.opinion)
                users[user_idx] = user
                pages[page_idx] = page
                # re-compute the probability user stops and update global rate
                r = ratio(user)
                ups = (r * p1 + (1 - r) * p2)
                pstop[user_idx] = ups
                ppage[page_idx] += 1
                dR += ups
                if output:
                    print t, user.id, page.id
        elif ev == 1: # user stops
            user_idx = _randwpmf(pstop, 1, prng)
            user = users[user_idx]
            eR -= user.rate
            r = ratio(user)
            dR -= (r * p1 + (1 - r) * p2)
            user = None
            del users[user_idx]
            del pstop[user_idx]
            del pedit[user_idx]
        elif ev == 2: # new user
            o = prng.uniform()
            user = User(args.const_succ, args.const_succ, o, args.daily_edits)
            users.append(user)
            r = ratio(user)
            ups = (r * p1 + (1 - r) * p2)
            eR += user.rate
            dR += ups
            pstop.append(ups)
            pedit.append(user.rate)
        else: # new page
            if len(users):
                user_idx = prng.randint(0, len(users))
                user = users[user_idx]
                page = Page(args.const_pop + 1, user.opinion)
                pages.append(page)
                ppage.append(page.edits)
        if args.info_file is not None:
            args.info_file.write('%g %g %g\n' % (t, len(users), len(pages)))
    return num_events

