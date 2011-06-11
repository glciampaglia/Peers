import csv
import os
import sys
import numpy as np
from argparse import ArgumentParser, FileType
from itertools import groupby
from pprint import pprint

def main(args):
    ''' reads index, groups by parameters '''
    data_archive = np.load(args.data)
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(args.index.read(1000))
    args.index.seek(0)
    reader = csv.DictReader(args.index, dialect=dialect)
    parameters = reader.fieldnames[:-1]
    if args.error:
        outfields = parameters + [ 'average', 'error' ]
    else:
        outfields = parameters + [ 'average' ]
    writer = csv.DictWriter(sys.stdout, outfields, dialect=dialect)
    keyfunc = lambda row : tuple([ row[name] for name in parameters ])
    writer.writeheader()
    for k, subiter in groupby(reader, keyfunc):
        averages = []
        for row in subiter:
            fn, _ = os.path.splitext(row['file'])
            averages.append(data_archive[fn].mean())
        # averages can contains NaNs
        nans = np.isnan(averages)
        if nans.any():
            averages = averages[True - nans]
            if args.verbose:
                nn = np.sum(nans)
                n = len(isnans)
                print >> sys.stderr, 'error: %s:' % pprint(k)
                print >> sys.stderr, '  filtering %d/%d NaNs' % (nn, n)
        outrow = dict(row)
        del outrow['file']
        outrow['average'] = np.mean(averages)
        if args.error:
            outrow['error'] = np.std(averages) / np.sqrt(len(averages))
        writer.writerow(outrow)
     
def make_parser():
    parser = ArgumentParser(description='computes average lifetime')
    parser.add_argument('index', type=FileType('r'), help='index file')
    parser.add_argument('data', type=FileType('r'), help='data file')
    parser.add_argument('-e', '--error', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser

if __name__ == '__main__':
    ns = parser.parse_args()
    main(ns)
