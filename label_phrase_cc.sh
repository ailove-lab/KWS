#!/bin/bash
./build/label_phrase                         \
--graph=train/$1/${1}_frozen.pb              \
--labels=train/${1}/training/${1}_labels.txt \
--wav=$2
