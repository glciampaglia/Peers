# This file should not be used directly as a script. Rather, source it from the
# main simulation script

function simulate {
    # Bring ipcluster up
    if [[ $usessh = 1 ]]; then
        ipcluster ssh --clusterfile $clusterconf 2>&1 >$clusterlog &
        PID=$!
        echo -n "waiting for ipcluster ssh ($PID) to start..."
    else
        ipcluster local -n `cpuno` 2>&1 >$clusterlog &
        PID=$!
        echo "waiting for ipcluster local ($PID) to start..."
    fi
    sleep 5
    # Launch simulation
    peerstool jobs -r $reps "$1" < $sample | peerstool pexec -v
    # Take ipcluster down
    kill -2 $PID
    sleep 5
    kill -CONT $PID &>/dev/null # if kill returns 0 then process is still alive
    if [[ $? = 0 ]]; then
        echo "ipcluster($PID) did not terminate. Stop it manually."
    fi
}

function makeindex {
    # Create the index of output files
    tmpindex=`tempfile -p index`
    tmpsample=`tempfile -p sample`
    seq 0 $((size*reps-1)) | sed -e"s/.*/$prefix&.npy/" > $tmpindex
    script=$(cat <<EOF
import sys
for l in sys.stdin:
    for i in xrange($reps):
        print l,
EOF
)
    python -c "$script" < $sample > $tmpsample
    paste -d, $tmpsample $tmpindex > $index
    rm -f $tmpindex $tmpsample
}

function compress {
    if [[ $# = 0 ]] ; then return ; fi
    tar cvfz $prefix.tar.gz $@ >/dev/null
    TAR_ESTAT=$?
    if [[ $TAR_ESTAT = 0 ]] ; then
        echo "simulation saved in $prefix.tar.gz."
    else
        echo "error: tar exited with status $TAR_ESTAT."
    fi
}

function cpuno {
    if [[ -e /sys ]]; then
        cpus=`ls -1 /sys/devices/system/cpu/cpu* | grep -e [0-9]$ | wc -l`
    elif [[ -e /proc ]]; then
        cpus=`cat /proc/cpuinfo | grep -e ^processor | wc -l`
    else
        echo error: cannot establish number of cpus to use
        exit 1
    fi
    echo $cpus
}

function printhelp {
    cat <<EOF

parallel simulation script for \`peerstool' © 2011 Giovanni Luca Ciampaglia

usage: `basename $0` [options]

options are:

-h, --help              print this message and exit
-r NUM, --reps NUM      execute NUM repetitions
-p PREF, --prefix PREF  output file names will be prefixed with PREF
-s, --ssh               call the IPython cluster with ssh mode
-o, --overwrite         overwrite existing output file

will need the following files in the working dir:

sample.txt          sample file
defaults            fixed-value options for \`peerstool peer'
options             sample options for \`peerstool peer'
cluster_conf.py     (optional) configuration file for \`ipcluster ssh'

EOF
}
