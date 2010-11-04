#!/usr/bin/env python
# coding=utf-8

'''
reads a sequence of jobs from stdin and executes them on the cluster. jobs are
command lines that are executed in workers
'''

# file: jobs.py
# vim:ts=8:sw=4:sts=4

import os
import sys
from argparse import ArgumentParser, FileType
from warnings import catch_warnings, simplefilter
with catch_warnings():
    simplefilter('ignore', DeprecationWarning)
    from IPython.kernel.client import MultiEngineClient, TaskClient, CompositeError

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
    try:
        return p.communicate()
    finally:
        if p.returncode != 0:
            raise RuntimeError('process %d failed with return code %d' %
                    (p.pid, p.returncode), cmd)

desc = 'Reads a sequence of jobs from stdin and executes them on a cluster.'

def make_parser():
    parser = ArgumentParser(description=desc)
    parser.add_argument('-t', '--taskclient', action='store_true',
            help='Turn load-balancing and fault-tolerance on ', default=False)
    parser.add_argument('-v', '--verbose', action='store_true',
            help='Print list of commands', default=False)
    parser.add_argument('-f', '--from-file', dest='input_file',  metavar='file',
            type=FileType('r'), help='reads jobs list from file',
            default=sys.stdin)
    parser.add_argument('-D', '--debug', action='store_true', default=False,
            help='raise Python exceptions to the console')
    parser.add_argument('-w','--cwd', metavar='dir', default=os.getcwd(),
            help='set current working directory of engines to dir')
    return parser

def _setwd(args,client):
    client.execute('import os')
    client.execute('os.chdir(%s)' % repr(args.cwd))

def main(args):
    lines = ( l.strip() for l in iter(args.input_file.readline, '') )
    cmds = filter(lambda k : len(k), lines)
    if args.verbose:
        for i,c in enumerate(cmds):
            print '%d) "%s"' % (i,c)
    try:
        if args.taskclient:
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                tc = TaskClient()
            _setwd(args, tc)
            return tc.map(execpipe, cmds)
        else:
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                mec = MultiEngineClient()
            _setwd(args,mec)
            return mec.map(execpipe, cmds)
    except CompositeError,e:
        e.print_tracebacks()

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        ret = main(ns)
    except:
        ty,val,tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            name = ty.__name__
            print >> sys.stderr, '\n%s: %s\n' % (name, val)
    finally:
        if ns.input_file is not sys.stdin:
            ns.input_file.close()
