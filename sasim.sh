#!/bin/bash

# sasim.sh - simulation for sensitivity analysis

size=2 # sample size
reps=1 # repetitions
simtime=1 # in days
trantime=1 # in days

order=(daily_edits daily_users daily_pages confidence speed const_succ
 const_pop rollback_prob time_step)

echo ${order[@]} | sed -e 's/ /,/g' > params.txt

python lhd.py -i 1 100 -i 1 200 -i 1 200 -i 0 1 -i 0 0.5 -i 0 100 -i 0 100\
     -i 0 1 -i 0 1 $size 9 > sample.txt

# python winding.py -i 1 100 -i 1 200 -i 1 200 -i 0 1 -i 0 0.5 -i 0 100 -i 0 100\
#     -i 0 1 -i 0 1 $size 9 > sample.txt
# size=$(($size*${#order[@]})) # needed for the winding sampling

options="-e %(e)g -U %(U)g -P %(P)g -c %(c)g -s %(s)g --const-succ %(cs)g --const-pop %(cp)g --rollback-prob %(rp)g -t %(t)g"
sim_cmd="python cpeers.py $options -T $trantime $simtime"
lt_cmd="python lt.py -l out_%(count)s.npy"
ind_cmd="echo out_%(count)s.npy >> /tmp/index.txt"

source functions.sh

simulate "$sim_cmd | $lt_cmd" $reps 0
makeindex $size $reps
compress out*npy sample.txt index.txt params.txt

