#!/bin/bash
./build/label_wav                          \
--wav=$2                                   \
--graph=train/$1/frozen.pb                 \
--labels=train/$1/training/${1}_labels.txt \
--how_many_labels=1                        \
--input_name=wav_data                      \
--output_name=labels_softmax
