#!/bin/bash

NUM=10000

# daily_edits x confidence
peerstool effectplot peers/gsa/lifetime_lhd_50.txt 1 -e -i 0 3 -n $NUM\
    --labels daily_edits confidence lifetime -o de-co_$NUM.pdf -D

# daily_users x confidence
peerstool effectplot peers/gsa/lifetime_lhd_50.txt 1 -e -i 1 3 -n $NUM\
    --labels daily_users confidence lifetime -o du-co_$NUM.pdf -D

# daily_pages x confidence
peerstool effectplot peers/gsa/lifetime_lhd_50.txt 1 -e -i 2 3 -n $NUM\
    --labels daily_pages confidence lifetime -o dp-co_$NUM -D

# speed x confidence
peerstool effectplot peers/gsa/lifetime_lhd_50.txt 1 -e -i 4 3 -n $NUM\
    --labels speed confidence lifetime -o sp-co_$NUM.pdf -D

# rollback_prob x confidence
peerstool effectplot peers/gsa/lifetime_lhd_50.txt 1 -e -i 7 3 -n $NUM\
    --labels rollback_prob confidence lifetime -o rp-co_$NUM.pdf -D

# short_life x confidence
peerstool effectplot peers/gsa/lifetime_lhd_50.txt 1 -e -i 8 3 -n $NUM\
    --labels short_life confidence lifetime -o sl-co_$NUM.pdf -D

# long_life x confidence
peerstool effectplot peers/gsa/lifetime_lhd_50.txt 1 -e -i 9 3 -n $NUM\
    --labels long_life confidence lifetime -o ll-co_$NUM.pdf -D


