#!/bin/bash

# fitsim.sh - simulation for fitting model

# Simulate each value $reps time, for all $size values.
size=2 # sample size
reps=1 # repetitions

# Fix model parameters
baselife=100 # in days
dailyusers=50 # users/day
dailypages=10 # pages/day
dailyedits=5 # edits/day
timestep=$(echo 1/24|bc -l) # in days
simtime=5 # in days
trantime=5 # in days
cat > defaults <<EOF
-b
$baselife
-U
$dailyusers
-P
$dailypages
-e
$dailyedits
-t
$timestep
-T
$trantime
$simtime
EOF

# Define input sample
order=(confidence)
echo ${order[@]} | sed -e 's/ /,/g' > params.txt
step=$(echo 1/$size|bc -l)
LC_ALL=en seq $step $step 1 > sample.txt # for English locale for decimal sep

# Define simulator commands
sim_cmd="python cpeers.py -c %(c)g @defaults"
lt_cmd="python lt.py -lL out_%(count)s.npy" # store log-lifetime
ind_cmd="echo out_%(count)s.npy >> /tmp/index.txt"

source functions.sh

simulate "$sim_cmd | $lt_cmd" $reps 0
makeindex $size $reps
compress out*npy sample.txt index.txt params.txt defaults


