#!/bin/bash
./build/label_phrase               \
--wav=$1                           \
--graph=pretrained/gru_frozen.pb   \
--labels=pretrained/gru_labels.txt
