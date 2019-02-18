#!/bin/bash

./build/magne-kws                   \
--wav=validation/test.wav           \
--graph=pretrained/gru_frozen.pb    \
--labels=pretrained/gru_labels.txt  

