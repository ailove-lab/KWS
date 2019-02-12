#!/bin/bash

source config.sh

steps=500,500,500
rate=0.0005,0.0001,0.00002

function ds_cnn {
    model=ds_cnn
    out=train/$model
    python train.py                             \
    --model_architecture $model                 \
    --wanted_words=$words                       \
    --data_dir=$data_dir                        \
    --data_url=$data_url                        \
    --unknown_percentage=50                     \
    --model_size_info 6 276 10 4 2              \
                      1 276  3 3 2              \
                      2 276  3 3 1              \
                      1 276  3 3 1              \
                      1 276  3 3 1              \
                      1 276  3 3 1 1            \
    --dct_coefficient_count 10                  \
    --window_size_ms 40                         \
    --window_stride_ms 20                       \
    --learning_rate           $rate             \
    --how_many_training_steps $steps            \
    --summaries_dir $out/retrain_logs           \
    --train_dir $out/training
}

function crnn {
    model=crnn
    out=train/$model
    python train.py                             \
    --model_architecture $model                 \
    --wanted_words=$words                       \
    --data_dir=$data_dir                        \
    --data_url=$data_url                        \
    --unknown_percentage=50                     \
    --model_size_info 100 10 4 2 1 2 136 188    \
    --dct_coefficient_count 10                  \
    --window_size_ms   40                       \
    --window_stride_ms 20                       \
    --learning_rate           $rate             \
    --how_many_training_steps $steps            \
    --summaries_dir $out/retrain_logs           \
    --train_dir $out/training
}

function gru {
    model=gru
    out=train/$model
    python train.py                             \
    --model_architecture $model                 \
    --wanted_words=$words                       \
    --data_dir=$data_dir                        \
    --data_url=$data_url                        \
    --unknown_percentage=50                     \
    --model_size_info 1 400                     \
    --dct_coefficient_count 10                  \
    --window_size_ms 40                         \
    --window_stride_ms 20                       \
    --learning_rate           $rate             \
    --how_many_training_steps $stepes           \
    --summaries_dir $out/retrain_logs           \
    --train_dir $out/training
}

function lstm {
    model=lstm
    out=train/$model
    python train.py                             \
    --model_architecture=$model                 \
    --wanted_words=$words                       \
    --data_dir=$data_dir                        \
    --data_url=$data_url                        \
    --unknown_percentage=50                     \
    --model_size_info 188 500                   \
    --dct_coefficient_count 10                  \
    --window_size_ms 40                         \
    --window_stride_ms 20                       \
    --learning_rate           $rate             \
    --how_many_training_steps $steps            \
    --summaries_dir $out/retrain_logs           \
    --train_dir $out/training
}

lstm
crnn
ds_cnn
# gru
 
