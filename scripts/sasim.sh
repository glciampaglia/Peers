#!/bin/bash

# sasim.sh - simulation for sensitivity analysis

size=4 # sample size
reps=1 # repetitions
simtime=1 # in days
trantime=2 # in days

order=(daily_edits daily_users daily_pages confidence speed const_succ
 const_pop rollback_prob short_life long_life)

echo ${order[@]} | sed -e 's/ /,/g' > params.txt

peerstool lhd -i 1 100 -i 1 200 -i 1 200 -i 0 1 -i 0 0.5 -i 0 100 -i 0 100\
     -i 0 1 -i 0 1 -i 10 100 $size 10 > sample.txt

# peerstool winding -i 1 100 -i 1 200 -i 1 200 -i 0 1 -i 0 0.5 -i 0 100 -i 0 100\
#     -i 0 1 -i 0 1 -i 10 100 $size 10 > sample.txt
# size=$(($size*${#order[@]})) # needed for the winding sampling

options="-e %(e)g -U %(U)g -P %(P)g -c %(c)g -s %(s)g --const-succ %(cs)g --const-pop %(cp)g -r %(rp)g -l %(l)g -L %(L)g"
sim_cmd="peerstool peers -D --fast $options -T $trantime $simtime"
lt_cmd="peerstool lt -l out_%(count)s.npy"
ind_cmd="echo out_%(count)s.npy >> /tmp/index.txt"

source functions.sh

simulate "$sim_cmd | $lt_cmd" $reps 0
makeindex $size $reps
compress out*npy sample.txt index.txt params.txt cluster.log

