#!/usr/bin/env python
# coding=utf-8

'''
reads a sequence of jobs from stdin and executes them on the cluster. jobs are
command lines that are executed in workers
'''

# file: jobs.py
# vim:ts=8:sw=4:sts=4

import sys
from argparse import ArgumentParser
from warnings import catch_warnings, simplefilter
with catch_warnings():
    simplefilter('ignore', DeprecationWarning)
    from IPython.kernel.client import MultiEngineClient, TaskClient

def execpipe(cmd):
    '''
    executes a pipe command in subprocesses. Returns (stdout, stderr)
    '''
    import subprocess as sp
    cmdseq = cmd.split('|')
    prev_proc = None
    procs = []
    for c in cmdseq:
        if prev_proc is not None:
            p = sp.Popen(c.split(), stdin=prev_proc.stdout, stdout=sp.PIPE)
        else:
            p = sp.Popen(c.split(), stdout=sp.PIPE)
        procs.append(p)
        prev_proc = p
    return p.communicate()

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('-t', '--taskclient', action='store_true',
            help='Turn load-balancing and fault-tolerance on ', default=False)
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Print list of commands', default=False)
    return parser

def main(args):
    lines = ( l.strip() for l in iter(sys.stdin.readline, '') )
    cmds = filter(lambda k : len(k), lines)
    if args.verbose:
        for i,c in enumerate(cmds):
            print '%d) "%s"' % (i,c)
    if args.taskclient:
        with catch_warnings():
            simplefilter('ignore', DeprecationWarning)
            tc = TaskClient()
        tc.map(execpipe, cmds)
    else:
        with catch_warnings():
            simplefilter('ignore', DeprecationWarning)
            mec = MultiEngineClient()
        mec.map(execpipe, cmds)

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    main(ns)
