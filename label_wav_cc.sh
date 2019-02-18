#!/bin/bash
./build/label_wav                  \
--wav=data/запрещено/0_5_2215CC_0.wav  \
--graph=pretrained/gru_frozen.pb   \
--labels=pretrained/gru_labels.txt \
--input_name=wav_data              \
--output_name=labels_softmax
