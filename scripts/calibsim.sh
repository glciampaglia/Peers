#!/bin/bash

# fitsim.sh - simulation for calibration

# Simulate each value $reps time, for all $size values.
size=2 # sample size
reps=1 # repetitions
defaults=itwiki # file with defaults

# Define input sample
order=(const_pop const_succ confidence rollback_prob speed)
echo ${order[@]} | sed -e 's/ /,/g' > params.txt
step=$(echo 1/$size|bc -l)
LC_ALL=en seq $step $step 1 > sample.txt # for English locale for decimal sep

# Define simulator commands
sim_cmd="peerstool peers --fast -c %(c)g @$defaults"
lt_cmd="peerstool lt -lL out_%(count)s.npy" # store log-lifetime
ind_cmd="echo out_%(count)s.npy >> /tmp/index.txt"

source functions.sh

simulate "$sim_cmd | $lt_cmd" $reps 0
makeindex $size $reps
compress out*npy sample.txt index.txt params.txt defaults cluster.log


