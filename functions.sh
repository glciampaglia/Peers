function simulate {
    if [[ $# != 3 ]] 
    then
        echo "syntax: simulate CMD REPS USESSH"
        exit -1
    fi
    cmd=$1
    reps=$2
    usessh=$3
    if [[ ! -e sample.txt ]]
    then
        echo "sample.txt does not exist!"
        exit -1
    fi
    rm -f cluster.log &>/dev/null
    # Launch simulation
    if [[ $usessh = 1 ]]
    then
        ipcluster ssh --clusterfile cluster_conf.py 2>&1 >cluster.log &
    else
        ipcluster local -n 2 2>&1 >cluster.log &
    fi
    PID=$!
    echo -n "Waiting for ipcluster($PID) to start..."
    sleep 5
    python jobs.py -r $reps "$cmd" < sample.txt | python pexec.py -v
    kill -2 $PID
    sleep 2
    kill -CONT $PID &>/dev/null # if kill returns 0 then process is still alive
    if [[ $? = 0 ]]
    then
        echo "ipcluster($PID) did not terminate. Stop it manually."
    fi
}

function makeindex {
    if [[ $# != 2 ]]
    then
        echo 'syntax makeindex SIZE REPS'
        exit -1
    fi
    if [[ ! -e sample.txt ]]
    then
        echo "sample.txt does not exist!"
        exit -1
    fi
    size=$1
    reps=$2
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
}

function compress {
    if [[ $# = 0 ]]
    then
        echo "no files to compress!"
        exit -1
    fi
    files=${@}
    # Compress all output files plus index, sample, defaults and parameters list 
    tar cvfz out.tar.gz $files &>/dev/null
    TAR_ESTAT=$?
    if [[ $TAR_ESTAT = 0 ]]
    then
        echo "All files compressed. Removing intermediate files."
        rm -f $files
        echo "Simulation output stored in out.tar.gz."
    else
        echo "Error: tar exited with status $TAR_ESTAT."
        echo "No intermediate file has been removed."
    fi
}
