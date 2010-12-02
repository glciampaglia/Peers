#!/usr/bin/env python
# coding=utf-8

import os
import sys
import numpy as np
import datetime
import socket
from argparse import ArgumentParser, FileType

# TODO <Sat Nov 27 19:21:47 CET 2010>
# 1. write test cases

descr='Generate simulation commands from an input sample'

special_help='''
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
'''

def make_parser():
    parser = ArgumentParser(description=descr, fromfile_prefix_chars='@')
    parser.add_argument(
            'cmd',
            help='command line template',
            nargs='*',
            metavar='COMMAND')
    parser.add_argument('-c',
            '--cmds',
            nargs='+',
            type=FileType('r'), 
            metavar='FILE',
            help='read command template from %(metavar)s, produce pipe command')
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
            help='number of repetitions in input sample')
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
            help='input values separated by %(metavar)s (default: \'%(default)s\')')
    parser.add_argument(
            '-p',
            '--parameters',
            nargs='+',
            help='define parameter names (see --parameters-help)',
            metavar='NAME')
    parser.add_argument(
            '-H',
            '--parameters-help',
            action='store_true',
            help='more info on special parameter names you can use in command templates')
    return parser

def decorate(args, pointsiter, name, decorator):
    ''' decorate pointsiter with special parameter values '''
    args.parameters.remove(name)
    args.parameters.append(name)
    return ( decorator(point) for point in pointsiter )

def main(args):
    if args.parameters_help:
        print >> sys.stderr, special_help
        return
    cmds = args.cmd 
    if args.cmds is not None:
        cmds.extend([ c.readline().strip() for c in args.cmds ])
    cmdline = ' | '.join(cmds)
    if args.input.isatty():
        print >> sys.stderr, 'Reading sample from standard input ...'
    pointsiter = ( tuple(map(eval, line.strip().split(args.separator)))
            for line in iter(args.input.readline, '') )
    if args.parameters is not None:
        if 'num' in args.parameters:
            pointsiter = decorate(args, enumerate(pointsiter), 'num', 
                    lambda k : k[1] + (k[0],) )
        if 'date' in args.parameters:
            d = datetime.date.today()
            pointsiter = decorate(args, pointsiter, 'date', 
                    lambda k : k + (d,))
        if 'user' in args.parameters:
            user = os.environ['USER']
            pointsiter = decorate(args, pointsiter, 'user', 
                    lambda k : k + (user,))
        if 'host' in args.parameters:
            host = socket.gethostname()
            pointsiter = decorate(args, pointsiter, 'host', 
                    lambda k : k + (host,))
        if 'time' in args.parameters:
            func = lambda k : k + (datetime.datetime.now().time(),)
            pointsiter = decorate(args, pointsiter, 'time', func)
        if 'rep' in args.parameters:
            if not args.repetitions:
                raise ValueError(
                        'must specify a number of repetitions (-r/--repetitions)')
            func = lambda k : k[1] + (k[0] % args.repetitions,)
            pointsiter = decorate(args, enumerate(pointsiter), 'rep', func)
        pointsiter = ( dict(zip(args.parameters, point)) for point in pointsiter)
    for point in pointsiter:
        print cmdline % point

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
            name = ty.__name__
            parser.error('%s : %s' % (name, val))

