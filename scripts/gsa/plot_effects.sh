#!/bin/bash

NUM=10000
FILE=lifetime_lhd_50.txt

echo main effect plot
peerstool effectplot $FILE 1 -e -m -n $NUM -o main_$NUM.pdf\
    -p params.txt

echo daily_edits x confidence
peerstool effectplot $FILE 1 -e -i 0 3 -n $NUM\
    --labels daily_edits confidence lifetime -o de-co_$NUM.pdf -D

echo daily_users x confidence
peerstool effectplot $FILE 1 -e -i 1 3 -n $NUM\
    --labels daily_users confidence lifetime -o du-co_$NUM.pdf -D

echo daily_pages x confidence
peerstool effectplot $FILE 1 -e -i 2 3 -n $NUM\
    --labels daily_pages confidence lifetime -o dp-co_$NUM.pdf -D

echo speed x confidence
peerstool effectplot $FILE 1 -e -i 4 3 -n $NUM\
    --labels speed confidence lifetime -o sp-co_$NUM.pdf -D

echo rollback_prob x confidence
peerstool effectplot $FILE 1 -e -i 7 3 -n $NUM\
    --labels rollback_prob confidence lifetime -o rp-co_$NUM.pdf -D

echo short_life x confidence
peerstool effectplot $FILE 1 -e -i 8 3 -n $NUM\
    --labels short_life confidence lifetime -o sl-co_$NUM.pdf -D

echo long_life x confidence
peerstool effectplot $FILE 1 -e -i 9 3 -n $NUM\
    --labels long_life confidence lifetime -o ll-co_$NUM.pdf -D

