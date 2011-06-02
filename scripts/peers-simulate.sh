#!/bin/bash

# Simulation script
#
# author: Giovanni Luca Ciampaglia
# email: ciampagg@usi.ch

# TODO <gio 26 mag 2011, 10.00.34, CEST>: 
# 1. needs to put a trap for ^C during simulation
# 2. test simulation returned correctly otherwise exit script w/ error code

# functions
source functions.sh

# configuration parameters
source config.sh

# test we have the right implementation of getopt
getopt -T
if [ $? != 4 ] ; then echo 'error: needs enhanced getopt from util-linux-ng' ; exit 1 ; fi

# parse command line options with getopt
TEMP=`getopt -u -n $(basename $0) -o $SHORT_OPTS -l $LONG_OPTS -- "$@"`
if [ $? != 0 ] ; then exit 1 ; fi

# set positional arguments to the parsed string
eval set -- "$TEMP"

# update variables with value from positional arguments
while true; do
    case "$1" in
        --) 
            shift ; break ;;
        -r|--reps) 
            reps=$2 ; shift 2 ;;
        -p|--prefix)
            prefix=$2 ; shift 2 ;;
        -c|--cpus)
            cpus=$2 ; shift 2 ;;
        -s|--ssh)
            usessh=1 ; shift ;;
        -o|--overwrite)
            overwrite=1 ; shift ;;
        -h|--help)
            printhelp ; shift ; exit 0 ;;
    esac;
done

# test that all input files exist
inputfiles=( $defaults $options $sample $params $clusterconf )
for file in ${inputfiles[*]}; do
    if [[ ! -e $file ]]; then
        echo error: file $file is missing.
        exit 1
    fi
done

# test we don't overwrite output file
if [[ $overwrite = 0 && -e $prefix.tar.gz ]] 
then 
    echo "error: output file $prefix.tar.gz already exists."
    exit 2
fi

# define other simulation parameters 
size=`wc -l < $sample`
options=`<$options`

# define commands to be executed on the cluster
outfile="$prefix%(count)s.npy"
sim_cmd="peerstool peers --fast $options @$defaults"
lt_cmd="peerstool lt -lL $outfile" # store log-lifetime

# GO!
simulate "$sim_cmd | $lt_cmd"
makeindex
# store all array files in an NPZ file
zip -q $data $prefix*.npy && rm -f $prefix*.npy
compress $data $index $defaults $clusterlog && rm -f $index $data

