#!/bin/bash

# This script converts simulation archives to the new format used in
# peers-simulate.sh. The new format uses an NPZ archive to store all simulation
# files, and the index file is a header line listing output variables names.

prog=`basename $0`

# test we have all utilities
for i in mktemp tar zip 
do
    [[ ! `which $i` ]] && { echo "$prog: $i utility missing. Please install it" ; exit 0 ; }
done

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

for i in $@
do
    convert $i 
done
