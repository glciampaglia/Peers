#!/bin/sh

num=10
file=prova
python jobs.py 'python cpeers.py' -p 'python lt.py' -s defaults\
    -S ltdefaults -d confidence -d rollback-prob -r 0 1 -r 0 1 -o $file lhd $num\
    | python cluster.py -v

if [ "$?" = 0 ];
then
    zip $file.npz $file-*.npy "$file"_index.npy && rm -f $file-*.npy "$file"_index.npy
else
    echo 'failed'
fi
