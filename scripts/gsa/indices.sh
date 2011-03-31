#!/bin/bash
FILE=lifetime_lhd_50.txt
PARAMS=params.txt
NUM=10000

echo PCC and scatter plot
peerstool pcc -o scatter.pdf -e -p $PARAMS $FILE > pcc.txt

echo main and total interaction effect indices
peerstool vardec $FILE 1 -e -p $PARAMS -r $NUM > effects.txt

echo standardized regression coefficients
peerstool regr -p $PARAMS -e $FILE > regr.txt
