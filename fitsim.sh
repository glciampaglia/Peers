#!/bin/bash

# fitsim.sh - simulation for fitting model

# Simulate each value $reps time, for all $size values.
size=25 # sample size
reps=10 # repetitions

# Fix model parameters
baselife=100 # in days
dailyusers=10 # users/day
dailypages=1 # pages/day
dailyedits=5 # edits/day
timestep=$(echo 1/24|bc -l) # in days
simtime=365 # in days
trantime=365 # in days
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
sim_cmd="python cpeers.py -e %(e)g @defaults"
lt_cmd="python lt.py -lL out_%(count)s.npy" # store log-lifetime
ind_cmd="echo out_%(count)s.npy >> /tmp/index.txt"

# Launch simulation
# ipcluster ssh --clusterfile cluster_conf.py 2>&1 >cluster.log &
rm -f cluster.log &>/dev/null
ipcluster local -n 2 2>&1 >cluster.log &
PID=$!
echo -n "Waiting for ipcluster($PID) to start..."
sleep 5
python jobs.py -r $reps "$sim_cmd | $lt_cmd" < sample.txt | python pexec.py -v
kill -2 $PID
sleep 2
kill -CONT $PID &>/dev/null # if kill returns 0 then process is still alive
if [[ $? = 0 ]]
then
    echo "ipcluster($PID) did not terminate. Stop it manually."
fi

# Create the index of output files
seq 0 $(($size*$reps-1)) | sed -e's/.*/out_&.npy/' > /tmp/index.txt
rep_script=$(cat <<EOF
import sys
for l in sys.stdin.readlines():
    for i in xrange($reps):
        print l,
EOF
)
python -c "$rep_script" < sample.txt > /tmp/sample.txt
paste -d, /tmp/sample.txt /tmp/index.txt > index.txt
rm -f /tmp/{index,sample}.txt

# Compress all output files plus index, sample, defaults and parameters list 
tar cvfz out.tar.gz out*npy sample.txt index.txt params.txt defaults &>/dev/null
TAR_ESTAT=$?
if [[ $TAR_ESTAT = 0 ]]
then
    echo "All files compressed. Removing intermediate files."
    rm -f out*.npy {sample,index,params}.txt
    echo "Simulation output stored in out.tar.gz."
else
    echo "Error: tar exited with status $TAR_ESTAT."
    echo "No intermediate file has been removed."
fi

