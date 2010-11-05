#!/bin/sh

# Launches a simulation using jobs.py/cluster.py and an IPython.kernel cluster
#
# You first need to setup a cluster. 'ipcluster local' is the easiest way to do
# so. Then write a file with parameter values for jobs.py (e.g. jobs.defs, see
# jobs.py -h to see the syntax that is accepted by the script). Next, choose the
# number of repetitions and the file name for the archive that will store the
# simulated data. If you put 'foo', then 'foo.npz' will be created, with data
# files 'foo-0.npy', 'foo-1.npy', ... etc, plus metadata stored in
# 'foo_index.npy' and 'foo_defaults.npy'. You can access these files with
# NumPy's 'load' function. If 'foo.npz' already exists, the script will refuse
# to continue.
# 
# The simulation will be launched in the current working directory. All
# intermediate output files will be wiped out at the end of the simulation.

# test arguments
if [ $# != 3 ];
then
    echo "usage $0 NUM OUTPUT CONFIG"
    exit 0
fi

# get arguments
num=$1
file=$2
conf=$3

# check if output exists and in case refuse to continue
if [ -e "$file.npz" ];
then
    echo "file $file.npz exists."
    exit 0
fi

# figure out where the scripts are
me=`which $0`
bindir=`dirname $me`

# print banner
echo '********************************************************************************'
date
echo '********************************************************************************'

# launch the simulations
python $bindir/jobs.py "python $bindir/cpeers.py" -p "python $bindir/lt.py" \
    -o $file @$conf | python $bindir/cluster.py -v -w `pwd`

# test exit status
eval=$?
if [ $eval = 0 ];
then
    zip $file.npz $file[-_]*.npy && rm -f $file[-_]*.npy 
else
    echo "simulation failed with code $eval"
    exit 1
fi

# print final banner
echo '********************************************************************************'
date
echo '********************************************************************************'

