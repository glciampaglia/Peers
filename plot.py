#!/usr/bin/env python
# coding=utf-8
# file: plot.py
# vim:ts=8:sts=4:sw=4

import numpy as np
import matplotlib.pyplot as pp
from argparse import ArgumentParser, FileType, Action
from itertools import groupby

def _figure(args):
    fig = pp.figure(figsize=args.figsize)
    ax = fig.add_subplot(111)
    if args.xlabel:
        ax.set_xlabel(args.xlabel, fontsize=args.font_size)
    else:
        ax.set_xlabel(args.name, fontsize=args.font_size)
    if args.ylabel:
        ax.set_ylabel(args.ylabel, fontsize=args.font_size)
    return (ax,fig)

def _mpl_kwargs(args):
    return dict(c=args.color, ls=args.line_style, marker=args.marker,
            ms=args.marker_size)

def _dataiter(args):
    for i in xrange(len(args.input.files)-2):
        yield args.input[args.name+'-%d' % i]

def plotcmd(args):
    ''' plot the mean of data, optionally grouping by the same x-value '''
    data = np.asarray( zip(args.index[args.dimension],
            [ np.mean(d) for d in _dataiter(args)]) )
    ax, fig = _figure(args)
    if args.grouped:
        gdata = []
        for k, subiter in groupby(data, lambda k : k[0]):
            subdata = list((d for k,d in subiter))
            n = len(subdata)
            gdata.append((k, np.mean(subdata), np.std(subdata)/np.sqrt(n)))
        data = np.asarray(gdata)
        ax.errorbar(data[:,0], data[:,1], data[:,2]/2., **_mpl_kwargs(args))
    else:
        l, = ax.plot(data[:,0], data[:,1], **_mpl_kwargs(args))
    pp.draw()
    if args.output:
        for fmt in args.formats:
            pp.savefig(args.output+'.'+fmt)
    pp.show()
    return data

def listcmd(args):
    ''' display information on data '''
    set_index = set(map(tuple, args.index))
    cmd = ' '.join(map(lambda k : ' '.join(map(str,k.item())), args.defaults))
    print 'NAME: %s' % args.name
    print 'TOTAL: %d simulations' % (len(args.input.files)-2)
    print 'PARAMETERS: %s' % ', '.join(args.index.dtype.names)
    print 'REALIZATIONS: %d per parameter ndpoint' % (len(args.index)/len(set_index))
    if args.print_index:
        print 'INDEX:'
        for i,point in enumerate(set_index):
            print '  (%d)\t' % i + repr(point)
    if args.print_defaults:
        print 'DEFAULTS: %s' % cmd

class NumPyLoad(Action):
    def __call__(self, parser, ns, values, option_string=None):
        setattr(ns, self.dest, np.load(values[0]))

class MakeTuple(Action):
    def __call__(self, parser, ns, values, option_string=None):
        setattr(ns, self.dest, tuple(values))

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('input', help='read data from file', action=NumPyLoad,
            type=FileType('r'), nargs=1, metavar='INPUT')
    subparsers = parser.add_subparsers()
# list contents
    list_parser = subparsers.add_parser('list', help='List data file contents')
    list_parser.set_defaults(func=listcmd)
    list_parser.add_argument('--print-index', action='store_true')
    list_parser.add_argument('--print-defaults', action='store_true')
# plot first moment of data
    plot_mean_parser = subparsers.add_parser('plot_mean', 
            help='Plot data along one dimension of the parameter space')
    plot_mean_parser.add_argument('dimension', help='dimension name', 
        metavar='DIMENSION',)
#    plot_mean_parser.add_argument('key_dimension', metavar='KEY DIM', nargs='?',
#            help='will produce one plot per each value along %(metavar)s')
    plot_mean_parser.add_argument('-s', '--save', help='save plot data to file', 
            metavar='file', type=FileType('w'))
    plot_mean_parser.add_argument('-o', '--output', help='save plots to %(metavar)s.FMT',
            metavar='FILE')
    plot_mean_parser.add_argument('-f', '--formats', nargs='+', metavar='FMT', 
            default=['png'], help='graphics output format (default: %(default)s)')
    plot_mean_parser.add_argument('--grouped', action='store_true')
    plot_mean_parser.set_defaults(func=plotcmd)
# matplotlib arguments
    plot_mean_parser.add_argument('--figsize', nargs=2, default=None, type=float,
            action=MakeTuple, metavar='h w')
    plot_mean_parser.add_argument('--marker-size', default=8, type=int, metavar='size')
    plot_mean_parser.add_argument('--line-style', default=' ', metavar='style')
    plot_mean_parser.add_argument('--marker', default='o', metavar='type')
    plot_mean_parser.add_argument('--xlabel', metavar='text', 
            help='x axis label caption (escape TeX macros with \\\\)')
    plot_mean_parser.add_argument('--ylabel', metavar='text', 
            help='y axis label caption (escape TeX macros with \\\\)')
    plot_mean_parser.add_argument('--legend-location', type=int, metavar='loc',
            help='see matplotlib.pyplot.legend\'s docstring')
    plot_mean_parser.add_argument('--font-size', default=18, type=int,
            metavar='size', help='font size')
    plot_mean_parser.add_argument('--color', default='k', help='color',
            metavar='color')
    return parser

def check_parser(args, parser):
    name = args.input.files[0].split('-')[:-1]
    args.name = ''.join(name)
    args.index = args.input[args.name+'_index']
    args.defaults = args.input[args.name+'_defaults']

def main(args):
    ret = args.func(args)
    if args.save:
        np.save(args.save, ret)

if __name__ == '__main__':
    parser = make_parser()
    ns = parser.parse_args()
    check_parser(ns, parser)
    main(ns)
