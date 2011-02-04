#!/bin/bash

# sasim.sh - simulation for sensitivity analysis

size=50 # sample size
reps=3 # repetitions

order=(daily_edits daily_users daily_pages confidence speed const_succ
 const_pop rollback_prob)

echo ${order[@]} | sed -e 's/ /,/g' > params.txt

# python lhd.py -i 1 100 -i 1 200 -i 1 200 -i 0 1 -i 0 0.5 -i 0 100 -i 0 100\
#     -i 0 1 $size 8 > sample.txt
python winding.py -i 1 100 -i 1 200 -i 1 200 -i 0 1 -i 0 0.5 -i 0 100 -i 0 100\
    -i 0 1 $size 8 > sample.txt

options="-e %(e)g -U %(U)g -P %(P)g -c %(c)g -s %(s)g --const-succ %(cs)d --const-pop %(cp)d --rollback-prob %(rp)g"
step=$(echo 1/24|bc -l)
sim_cmd="python cpeers.py $options -T 1 -t $step 1"
lt_cmd="python lt.py -l out_%(count)s.npy"
ind_cmd="echo out_%(count)s.npy >> /tmp/index.txt"

# # With xargs
# python jobs.py -r $reps "$sim_cmd | $lt_cmd" < sample.txt \
#    | xargs -I{} sh -c "{}"

# With IPython Kernel
#ipcluster ssh --clusterfile cluster_conf.py 2>&1 >cluster.log &
ipcluster local -n 2 2>&1 >cluster.log &
PID=$!
echo -n "Waiting for ipcluster($PID) to start..."
sleep 6
python jobs.py -r $reps "$sim_cmd | $lt_cmd" < sample.txt | python pexec.py -v
kill -2 $PID
sleep 2
if [[ -n $(ps ax | grep $PID) ]]
then
    echo "ipcluster($PID) did not terminate. Stop it manually."
fi

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
tar cvfz out.tar.gz out*npy sample.txt index.txt params.txt &>/dev/null
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

