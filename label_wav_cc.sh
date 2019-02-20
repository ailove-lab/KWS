#!/bin/bash
./build/label_wav                  \
--wav=$1                           \
--graph=pretrained/gru_frozen.pb   \
--labels=pretrained/gru_labels.txt \
--how_many_labels=1                \
--input_name=wav_data              \
--output_name=labels_softmax
