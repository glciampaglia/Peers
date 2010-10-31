#!/bin/sh

if [ $# -ne 2 ];
then
    echo "usage: $0 N FILE"
    exit 2
fi

n=$1
output="$2"
for i in `seq 1 $n` 
do 
    echo -n "python peers.py @defaults | python lt.py -l $output-$i.npy\n"
done
