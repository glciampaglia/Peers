#!/usr/bin/env python
# coding=utf-8

'''
jobs.py -- A template-based command line builder.

Introduction
------------
jobs.py lets you create a list of command lines from a template specification
that you pass as input.

You can pipe it to xargs in order to execute. A very simple example is the
following:

    ::

    $> seq 10 | jobs.py 'echo "%d) hello, world!"' | xargs -l 

Of course you get the same effect with a simpler:

    ::

    $> seq 10 | xargs -l -i% echo '%) hello, world!'

But jobs.py lets you create more complex patterns. In general, you specify a
command-line template using the Python string formatting syntax (e.g. "%d" % 1
-> "1"), which makes it a very flexible tool.

File redirections
-----------------

The simplest way to execute commands produced by jobs.py is to xargs, but to get
it working with commands that use file redirection you must pass the command
lines to a shell.  For example, this WILL NOT work:

    :: 
    
    $> jobs.py 'peers.py %d %d > output' < sample | xargs -l1 python 

Because '> output' will be passed to python as an argument for peers.py. This
will work instead:

    ::
    
    $> jobs.py 'python peers.py %d %d > output' < sample | xargs -l1 -i% sh -c "%"

Because the shell will interpret '> output' as a file redirection.
On the other hand, if you have an IPython cluster running, you can use instead
pexec.py, which does that automatically.
'''

import re
import os
import sys
import datetime
import socket
from argparse import ArgumentParser, FileType

# TODO <Sat Nov 27 19:21:47 CET 2010>
# 1. write test cases

descr='Command line builder'

extended_help='''
A number of parameter names you can pass to -p/--parameters are interpreted as
special parameters. The script will recognize this and add additional
information to the input sample points. This can help in some situations when
the commands you want to execute must contain additional information to the
values of the input sample points.

Note that when you declare any special parameter you must also declare all other
parameter names and use the corresponding syntax for string substitution in your
command templates, i.e., you cannot mix '%g' and %(param)s in your command
templates.

Special parameter names you can use in command templates are:

'num'   - job number. 
          Useful for redirecting to numbered output files, e.g.: 
            
        python lt.py -l > output-%(num)d

'rep'   - repetition index

'date'  - the current date as returned by datetime.date.today()

'user'  - user's name, taken from the shell environment

'host'  - host name, taken with (2) gethostname

'time'  - the current time stamp as returned by datetime.datetime.now()

'count' - a progressive counter
'''

# http://docs.python.org/release/2.6.6/html/library/stdtypes.html#string-formatting-operations 
pat = re.compile('%(?:\(.+?\))?[- #0+]*(\d*\.\d*)?[diouxXeEfFgGcrs%]')

def make_parser():
    parser = ArgumentParser(description=descr, fromfile_prefix_chars='@')
    parser.add_argument(
            'cmd',
            nargs='?',
            help='command line template',
            metavar='COMMAND')
    parser.add_argument('-f',
            '--file',
            type=FileType('r'), 
            metavar='FILE',
            dest='cmd_file',
            help='read command from %(metavar)s instead of command line')
    parser.add_argument(
            '-D',
            '--debug',
            action='store_true', 
            help='raise Python exceptions to the console')
    parser.add_argument(
            '-r',
            '--repetitions',
            type=int,
            metavar='NUM',
            default=1,
            help='repeat each line of input sample %(metavar)s times.')
    parser.add_argument(
            '-i',
            '--input',
            type=FileType('r'),
            default=sys.stdin,
            metavar='FILE',
            help='read input sample from %(metavar)s')
    parser.add_argument('-s',
            '--separator',
            default=',',
            metavar='CHAR', 
            help='input values are separated by %(metavar)s (default: \'%(default)s\')')
    parser.add_argument(
            '-H',
            '--extended-help',
            action='store_true',
            help='more info on special parameter names you can use in command templates')
    return parser

cnt = 0

def _repl(m):
    ''' replacement function. For internal use only. '''
    global cnt
    fmt = m.group()[1:]
    r = '%%(param-%d)%s' % (cnt,fmt)
    cnt += 1
    return r

def normalize(cmd):
    '''
    transform all interpolation format specifiers into named form, e.g.

    :: 
    
        %d becomes %(param-0)d
    '''
    global cnt
    pat = re.compile('%[- #0+]*(\d*\.\d*)?[diouxXeEfFgGcrs%]')
    norm_cmd = pat.sub(_repl, cmd)
    cnt = 0
    return norm_cmd

def params(cmd):
    ''' extract parameter names from normalized specifiers '''
    pat = re.compile('\((?P<name>.+?)\)')
    return pat.findall(cmd)

def decorate(parameters, valuesiter, name, decorator):
    ''' decorate valuesiter with special parameter values '''
    parameters.remove(name)
    parameters.append(name)
    return ( decorator(value) for value in valuesiter )

def rep(valuesiter, n):
    for line in valuesiter:
        for i in xrange(n):
            yield line

def main(args):
    if args.extended_help:
        print >> sys.stderr, extended_help
        return
    if args.cmd_file is not None:
        args.cmd = args.cmd_file.readline().strip()
    if args.cmd is None:
        raise ValueError('must provide a command template')
    args.cmd = normalize(args.cmd)
    args.parameters = params(args.cmd)
    if args.input.isatty():
        print >> sys.stderr, 'Reading from terminal. Press ^D to finish.'
    valuesiter = rep(( tuple(map(eval, line.strip().split(args.separator)))
            for line in iter(args.input.readline, '') ), args.repetitions )
    if 'num' in args.parameters:
        valuesiter = decorate(args.parameters, enumerate(valuesiter), 'num', 
                lambda k : k[1] + (k[0],) )
    if 'date' in args.parameters:
        d = datetime.date.today()
        valuesiter = decorate(args.parameters, valuesiter, 'date', 
                lambda k : k + (d,))
    if 'user' in args.parameters:
        user = os.environ['USER']
        valuesiter = decorate(args.parameters, valuesiter, 'user', 
                lambda k : k + (user,))
    if 'host' in args.parameters:
        host = socket.gethostname()
        valuesiter = decorate(args.parameters, valuesiter, 'host', 
                lambda k : k + (host,))
    if 'time' in args.parameters:
        func = lambda k : k + (datetime.datetime.now().time(),)
        valuesiter = decorate(args.parameters, valuesiter, 'time', func)
    if 'rep' in args.parameters:
        func = lambda k : k[1] + (k[0] % args.repetitions,)
        valuesiter = decorate(args.parameters, enumerate(valuesiter), 'rep', 
                func)
    if 'count' in args.parameters:
        func = lambda k : k[1] + (k[0],)
        valuesiter = decorate(args.parameters, enumerate(valuesiter), 'count',
                func)
    valuesiter = ( dict(zip(args.parameters, value)) for value in valuesiter)
    for value in valuesiter:
        print args.cmd % value

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    try:
        main(ns)
    except:
        ty,val,tb = sys.exc_info()
        if ns.debug:
            raise ty, val, tb
        else:
            if ty is KeyboardInterrupt:
                print
                sys.exit(1)
            name = ty.__name__
            parser.error('%s : %s' % (name, val))

