#!/bin/bash
DATASET=info_mmlhd_50

if [[ ! -e counts_$DATASET.npz ]];
then
    echo computing counts data in data/$DATASET. This may take some time.
    peerstool diagnostics counts data/$DATASET/info_*.txt -f 1 -o counts_$DATASET.npz
fi

echo 'transient time plot'
peerstool diagnostics plot counts_$DATASET.npz -u -m -r 365 -n 730 -a .2 -o images/transient.pdf
