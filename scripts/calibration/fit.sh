#!/bin/bash

function printhelp { 
    cat <<EOF

usage: $prog [ OPTIONS ] [-d DATA | --data DATA] DIR [DIR ...]

options:
 -h/--help  print this message and exits
 -c/--comp  number of components to use (default: "2 3")
 -d/--data  data directory ( *required* )
 -n/--dry   do not execute commands; print them instead

EOF
}


prog=`basename $0`
components="2 3"
temp=`getopt -o hc:d:n -l help,comp:data:dry -- "$@"`
data=''
dry=0

eval set -- "$temp"

while `true` ; do
    case "$1" in 
        --) shift 1 ; break ;; 
        -h|--help) printhelp ; exit 1 ;; 
        -c|--comp) components="$2" ; shift 2 ;; 
        -d|--data) data="$2" ; shift 2 ;;
	-n|--dry) dry=1 ; shift 1 ;;
    esac
done

if [[ $data = '' ]] ; then 
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
    for c in $components ; do
        for t in -t \  ; do
	    opts="`<bounds` `<weights_$c` -p5 -c$c $t fit -B"
            if [[ $t = ' ' ]] ; then 
                simulations="$d"_gmm_"$c".txt 
            else
                simulations="$d"_gmm_truncated_"$c".txt 
            fi
	    if [[ $dry = 1 ]] ; then
		echo peerstool calibration $opts $data/$d.npy $simulations 
	    else
		peerstool calibration $opts $data/$d.npy $simulations 
	    fi
        done
    done
    popd &> /dev/null
done

