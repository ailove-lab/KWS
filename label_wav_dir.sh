#!/bin/bash

rm results.txt
rm results/*

for m in $(echo Pretrained_models/*/*.pb)
do
    echo $m
    while read l 
    do
        n=$(basename -- $m)
        o=results/${n%.*}_${l}_results.txt        
	python label_wav_dir.py               \
        --graph=$m                            \
        --labels=Pretrained_models/labels.txt \
        --wav_dir=../data/dataset/$l > $o
	echo $m + $l: $(cat $o | grep $l | wc -l) / $(ls ../data/dataset/$l | wc -l) >> results.txt
    done < Pretrained_models/labels.txt
done
