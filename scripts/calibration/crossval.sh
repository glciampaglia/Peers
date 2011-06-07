#!/bin/bash

prog=`basename $0`

function printhelp {
    cat<<EOF
usage: $prog [-h] dir1 dir2 ...

descends into dir1, dir2, .. and performs cross-validation. Will look for
for reduced simulation data in files of the form:

    DIR_gmm_[_truncated_]NUM.txt

where DIR is in turn dir1, dir2, etc, and NUM is any single value passed
with option -c/--comp.

options:
 -h/--help           print this message and exit
 -c NUM/--comp NUM   number of GMM components (accepts multiple values)
 -t/--truncated      also use truncated GMM 

EOF
    exit 0
}

components="2 3"
truncated=0

temp=`getopt -n $prog -o hc:t -l help,comp:,truncated -- "$@"`

eval set -- "$temp"

while `true` ; do
    case "$1" in 
        -h|--help) printhelp ; shift 1 ;;
        -t|--truncated) truncated=1 ; shift 2 ;;
        -c|--comp) components=$2 ; shift 2 ;;
        --) shift 1 ; break ;;
    esac
done

if [[ $# = 0 ]] ; then
    printhelp
fi

args="$@"

for dir in $args ; do
    if [[ ! -d $dir ]] ; then
        echo "$prog: error: no such directory $dir"
    fi
done

for wiki in $@ ; do
    pushd $wiki
    for file in bounds index.txt data.npz ; do
        if [[ ! -e $file ]] ; then 
            echo $prog: error: please provide $file
            exit 1
        fi
    done
    bounds=`<bounds`
    for c in $components ; do
        reduce=$wiki_gmm_$c
        peerstool calibration $bounds -i index.txt -p5 -c$c crossval "$wiki"_gmm_"$c".txt -o cv_gmm_$c.pdf | tee cv_gmm_$c.txt
        if [[ $truncated = 1 ]] ; then
            peerstool calibration $bounds -i index.txt -t -p5 -c$c crossval "$wiki"_gmm_truncated_"$c".txt -o cv_gmm_truncated_$c.pdf | tee cv_gmm_truncated_$c.txt
        fi
    done
    popd
done
