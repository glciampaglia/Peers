#!/bin/bash
counts=counts.npz

if [[ ! -e $counts ]];
then
    echo could not find data file $counts. 
    echo Please run \`peerstool diagnostic counts\` on you simulation data
    exit 2
fi

echo 'transient time plot'
peerstool diagnostics plot $counts -u -m -r 365 -n 730 -a .2 -o transient.pdf
