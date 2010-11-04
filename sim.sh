#!/bin/sh

num=20
file=tau_conf_roll
if [ -e "$file.npz" ];
then
    echo "file $file.npz exists."
    exit 0
fi
python jobs.py 'python cpeers.py' -p 'python lt.py' -s defaults -R 10\
    -S ltdefaults -d confidence -d rollback-prob -r 0 1 -r 0 1 -o $file lhd $num\
    | python cluster.py -v -w `pwd`
if [ "$?" = 0 ];
then
    zip $file.npz $file[-_]*.npy && rm -f $file[-_]*.npy 
else
    echo 'failed'
fi
