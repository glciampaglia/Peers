#!/bin/bash

function printhelp { 
    cat <<EOF

usage: $prog [ OPTIONS ] [-d | --data] DATA DIR [DIR ...]

options:
 -h/--help  print this message and exits
 -c/--comp  number of components to use (default: "2 3")
 -d/--data  data directory ( *required* )

EOF
}


prog=`basename $0`
components="2 3"
temp=`getopt -o hc:d: -l help,comp:data: -- "$@"`

eval set -- "$temp"

while `true` ; do
    case "$1" in 
        --) shift 1 ; break ;; 
        -h|--help) printhelp ; exit 1 ;; 
        -c|--comp) components=$1 ; shift 2 ;; 
        -d|--data) data=$1 ; shift 2 ;;
    esac
done

if [[ ! -v data ]] ; then 
    printhelp
    exit 1 
fi

if [[ $# = 0 ]] ; then 
    echo "$prog: error: need one or more directories" 
    exit 1 
fi

dir="$@"

for d in $dir ; do
    pushd $d &>/dev/null
    if [[ ! -e bounds ]] ; then
        echo "$prog: error: need bounds file"
        exit 1
    fi
    opts="`<bounds` -p5 -c$c $t -iindex.txt fit -B"
    for c in $components ; do
        for t in -t \  ; do
            if [[ $t = ' ' ]] ; then 
                simulations="$d"_gmm_"$c".txt 
            else
                simulations="$d"_gmm_truncated_"$c".txt 
            fi
            peerstool calibration $opts $data/$d.npy $simulations 
        done
    done
    popd &> /dev/null
done

