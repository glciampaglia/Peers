# This file should not be used directly as a script. Rather, source it from the
# main simulation script

function simulate {
    # Bring ipcluster up
    if [[ $usessh = 1 ]]; then
        ipcluster ssh --clusterfile $clusterconf 2>&1 >$clusterlog &
        PID=$!
        echo -n "waiting for ipcluster ssh ($PID) to start..."
    else
        [[ -n cpus ]] || cpus=`cpuno`
        ipcluster local -n $cpus 2>&1 >$clusterlog &
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
    tmpindex=`mktemp index.XXXXX`
    tmpsample=`mktemp sample.XXXXX`
    seq 0 $((size*reps-1)) | sed -e"s/.*/$prefix&.npy/" > $tmpindex
    header=`<$params`,file
    echo $header > $index
    replicate $reps < $sample > $tmpsample
    paste -d, $tmpsample $tmpindex >> $index
    rm -f $tmpindex $tmpsample
}

function compress {
    if [[ $# = 0 ]] ; then return ; fi
    # -h / --dereference tells tar to dump files being pointed by symlinks
    tar chvfz $prefix.tar.gz $@ >/dev/null
    TAR_ESTAT=$?
    if [[ $TAR_ESTAT = 0 ]] ; then
        echo "simulation saved in $prefix.tar.gz."
    else
        echo "error: tar exited with status $TAR_ESTAT."
    fi
}

function cpuno {
    if [[ -e /sys ]]; then
        cpus=`ls -d1 /sys/devices/system/cpu/cpu* | grep -e [0-9]$ | wc -l`
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
-c NUM, --cpus NUM      use NUM cpus (only with local IPython cluster)
-s, --ssh               call the IPython cluster with ssh mode
-o, --overwrite         overwrite existing output file

will need the following files in the working dir:

sample.txt          sample file
defaults            fixed-value options for \`peerstool peer'
options             sample options for \`peerstool peer'
cluster_conf.py     (optional) configuration file for \`ipcluster ssh'

EOF
}

function replicate {     
    [[ -n $1 ]] || { echo error: expecting an argument ; exit 1 ; }     
    num=$1
    if [[ $((num)) -le 0 ]] ; then
        echo error: not a number or negative value: $num
        exit  1
    fi
    read;     
    while [[ -n $REPLY ]] ; do
        for (( i=0 ; i<$num ; i++ )) ; do
            echo $REPLY;         
        done
        read     
    done; 
} 
