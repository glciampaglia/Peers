#!/bin/bash

# oneparam.sh - changes a single parameter

# Simulate each value $reps time, for all $size values.
reps=1 # repetitions
size=5 # sample size
name=confidence
param_min=0
param_max=0.5
if [[ ! $param_min < $param_max ]]
then
    echo "illegal range: min = $param_min, max = $param_max"
    exit 1
fi

simtime=3 # in days
trantime=7 # in days

# Fix model parameters
shortlife=$(echo 1/24|bc -l) # days
longlife=100 # days
dailyusers=10 # users/day
dailypages=10 # pages/day
dailyedits=10 # edits/day
constpop=50 
constsucc=50
cat > defaults <<EOF
-l
$shortlife
-L
$longlife
-U
$dailyusers
-P
$dailypages
-e
$dailyedits
--const-pop
$constpop
--const-succ
$constsucc
-T
$trantime
$simtime
EOF

# Save parameter name
echo $name > params.txt

# Define input sample
param_range=$(echo "$param_max - $param_min" | bc -l)
param_step=$(echo $param_range/$size | bc -l)
LC_ALL=en seq $param_min $param_step $param_max > sample.txt # for English locale for decimal sep

# Define simulator commands
sim_cmd="peerstool peers --fast -c %(c)g @defaults"
lt_cmd="peerstool lt -lL out_%(count)s.npy" # store log-lifetime
ind_cmd="echo out_%(count)s.npy >> /tmp/index.txt"

source functions.sh

simulate "$sim_cmd | $lt_cmd" $reps 0
makeindex $size $reps
compress out*npy sample.txt index.txt params.txt defaults cluster.log
