#!/bin/bash

python label_wav_dir.py                         \
    --graph=train/GRU2/frozen.pb                 \
    --labels=train/GRU2/training/gru_labels.txt  \
    --wav_dir=$1

