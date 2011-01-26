# coding=utf-8
# file: pexec.py
# vim:ts=8:sw=4:sts=4

import os
import sys
import datetime
from argparse import ArgumentParser, FileType
from warnings import catch_warnings, simplefilter
with catch_warnings():
    simplefilter('ignore', DeprecationWarning)
    from IPython.kernel.client import MultiEngineClient, TaskClient, CompositeError

def execcmd(cmd):
    '''
    Worker's function. Executes cmd in a subshell. Returns exit code from the
    subshell.
    '''
    # since we are on the worker, we must import modules locally
    import subprocess as sp  
    return sp.call(cmd, shell=True)

desc = 'Executes input commands on an IPython.kernel cluster'

def make_parser():
    parser = ArgumentParser(description=desc)
    parser.set_defaults(verbose=1)
    parser.add_argument(
            '-t',
            '--taskclient',
            action='store_true',
            help='use load-balancing, fault-tolerant mode.',
            default=False)
    parser.add_argument(
            '-v',
            '--verbose',
            action='store_const',
            const=2,
            help='print list of commands',)
    parser.add_argument(
            '-q',
            '--quiet',
            action='store_const',
            const=0,
            help='suppress all output',
            dest='verbose')
    parser.add_argument(
            '-i',
            '--input',
            metavar='FILE',
            type=FileType('r'),
            help='reads jobs list from %(metavar)s',
            default=sys.stdin)
    parser.add_argument(
            '-D',
            '--debug',
            action='store_true',
            default=False,
            help='raise Python exceptions to the console')
    parser.add_argument(
            '-w',
            '--cwd',
            metavar='DIR',
            default=os.getcwd(),
            help='set current working directory of engines to %(metavar)s')
    parser.add_argument(
            '-d',
            '--dry-run',
            action='store_true',
            help='run as normal except that no command is actually executed')
    return parser

def _setwd(args, client):
    client.execute('import os')
    client.execute('os.chdir(%s)' % repr(args.cwd))

def banner():
    print '*'*80
    print datetime.datetime.now()
    print '*'*80

def main(args):
    if args.input.isatty():
        print >> sys.stderr, 'Reading from terminal. Press ^D to finish.'
    lines = ( l.strip() for l in iter(args.input.readline, '') )
    cmds = filter(lambda k : len(k), lines)
    if args.verbose > 0:
        banner()
    if args.verbose > 1:
        for i,c in enumerate(cmds):
            print 'JOB %d) %s' % (i,c)
    try:
        if args.dry_run:
            return # does not open client interface
        if args.taskclient:
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                tc = TaskClient()
            _setwd(args, tc)
            return tc.map(execcmd, cmds)
        else:
            with catch_warnings():
                simplefilter('ignore', DeprecationWarning)
                mec = MultiEngineClient()
            _setwd(args,mec)
            return mec.map(execcmd, cmds)
    except CompositeError,e:
        e.print_tracebacks()
    finally:
        if args.verbose > 0:
            banner()

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
        if ns.input is not sys.stdin:
            ns.input.close()
