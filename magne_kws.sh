#!/bin/bash

./build/magne-kws                   \
--wav=$1                            \
--graph=pretrained/gru_frozen.pb    \
--labels=pretrained/gru_labels.txt  

