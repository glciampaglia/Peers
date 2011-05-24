#!/bin/bash

# Simulation script
#
# author: Giovanni Luca Ciampaglia
# email: ciampagg@usi.ch

# functions
source functions.sh

# configuration parameters
source config.sh

# user parameters, default values
prefix=out # output file will have this prefix
reps=1 # will perform these many runs for each line in the sample
usessh=0 # if 1, ipcluster will be called with `ssh` mode, else `local`
overwrite=0 # if 1, will overwrite the output file, if it exists

# define getopt's option strings
SHORT_OPTS=r:p:sho
LONG_OPTS=reps:,prefix:,ssh,help,overwrite

# test we have the right implementation of getopt
getopt -T
if [ $? != 4 ] ; then echo 'error: needs enhanced getopt from util-linux-ng' ; exit 1 ; fi

# parse command line options with getopt
TEMP=`getopt -u -n $(basename $0) -o $SHORT_OPTS -l $LONG_OPTS -- "$@"`
if [ $? != 0 ] ; then exit 1 ; fi

# set positional arguments to the parsed string
eval set -- "$TEMP"

while true; do
    case "$1" in
        --) 
            shift ; break ;;
        -r|--reps) 
            reps=$2 ; shift 2 ;;
        -p|--prefix)
            prefix=$2 ; shift 2 ;;
        -s|--ssh)
            usessh=1 ; shift ;;
        -o|--overwrite)
            overwrite=1 ; shift ;;
        -h|--help)
            printhelp ; shift ; exit 0 ;;
    esac;
done

# test input files exist
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

# blank output and log files
>$clusterlog
>$index

# define simulation parameters 
size=`wc -l < $sample`
options=`<$options`

# define command variables
outfile="$prefix%(count)s.npy"
sim_cmd="peerstool peers --fast $options @$defaults"
lt_cmd="peerstool lt -lL $outfile" # store log-lifetime
ind_cmd="echo $outfile >> $index"

simulate "$sim_cmd | $lt_cmd && $ind_cmd"
makeindex
compress $prefix*.npy $sample $index $params $defaults $clusterlog
rm -f $prefix*.npy $index

