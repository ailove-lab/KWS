#!/bin/bash

python label_wav_dir.py      \
--graph=$1/frozen.pb         \
--labels=$1/conv_labels.txt  \
--wav_dir=data_test/short_16

