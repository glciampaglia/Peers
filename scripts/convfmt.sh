#!/bin/bash

# This script converts simulation archives to the new format used in
# peers-simulate.sh. The new format uses an NPZ archive to store all simulation
# files, and the index file is a header line listing output variables names.

prog=`basename $0`

function printhelp
{
    cat <<EOF

usage: $prog [-h|--help] FILE [ FILE .. ]

options:
 -h/--help  print this message and exit

EOF
}

function convert
{ 
    echo -n converting $i..
    dir=`mktemp -d` 
    if [[ -e $dir ]]
    then
        tar xfz "$1" -C $dir
        pushd $dir &>/dev/null
        if [[ -e data.npz ]]
        then
            echo
            echo $prog: error: $1: already converted
            rm -rf $dir
            return
        fi
        zip -q data.npz *.npy && rm -f *.npy
        header=`<params.txt`
        echo $header,file | cat - index.txt > index.txt.1
        rm -f index.txt && mv index.txt.1 index.txt
        rm -f params.txt sample.txt
        tar cfz "$1" *  
        popd &>/dev/null
        mv "$1" "$1".orig && mv $dir/"$1" .
        echo done. Original file saved in "$1".orig
        rm -rf $dir
    else
        echo
        echo $prog: error: $1: could not create tempdir $dir
    fi
} 

# test we have all utilities
for i in mktemp tar zip 
do
    [[ ! `which $i` ]] && { echo "$prog: $i utility missing. Please install it" ; exit 0 ; }
done

# parse command line
temp=`getopt -o h -l help -- "$@"`

eval set -- "$temp"

while `true` ; do
    case "$1" in 
        -h|--help) printhelp ; exit 0 ;;
        --) shift 1 ; break ;;
    esac
done

if [[ $# = 0 ]] ; then
    echo
    echo "$prog: error: need one or more files"
    printhelp
fi

for i in $@
do
    convert $i 
done
