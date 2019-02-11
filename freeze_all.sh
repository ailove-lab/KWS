#!/bin/bash

source config.sh

# python freeze.py           \
# --model_architecture=gru   \
# --model_size_info 1 154    \
# --dct_coefficient_count=10 \
# --wanted_words=$words      \
# --clip_duration_ms=1000    \
# --sample_rate=16000        \
# --window_size_ms=40        \
# --window_stride_ms=40      \
# --checkpoint=$ckpt         \

function ds_cnn {
    model=ds_cnn
    ckpt=`ls -t train/$model/training/best/*.ckpt-*.meta | head -1`
    ckpt=${ckpt%.*}
    echo $ckpt

    python freeze.py                            \
    --model_architecture=$model                 \
    --wanted_words=$words                       \
    --model_size_info 6 276 10 4 2              \
                      1 276  3 3 2              \
                      2 276  3 3 1              \
                      1 276  3 3 1              \
                      1 276  3 3 1              \
                      1 276  3 3 1 1            \
    --dct_coefficient_count 10                  \
    --clip_duration_ms=1000                     \
    --window_size_ms 40                         \
    --window_stride_ms 20                       \
    --checkpoint=$ckpt                          \
    --output_file=train/$model/frozen.pb
}


function crnn {
    model=crnn
    ckpt=`ls -t train/$model/training/best/*.ckpt-*.meta | head -1`
    ckpt=${ckpt%.*}
    echo $ckpt

    python freeze.py                            \
    --model_architecture=$model                 \
    --wanted_words=$words                       \
    --model_size_info 100 10 4 2 1 2 136 188    \
    --dct_coefficient_count 10                  \
    --window_size_ms 40                         \
    --window_stride_ms 20                       \
    --checkpoint=$ckpt                          \
    --output_file=train/$model/frozen.pb
}


function gru {
    model=gru
    ckpt=`ls -t train/$model/training/best/*.ckpt-*.meta | head -1`
    ckpt=${ckpt%.*}
    echo $ckpt

    python freeze.py                            \
    --model_architecture $model                 \
    --wanted_words=$words                       \
    --data_dir=$data_dir                        \
    --unknown_percentage=50                     \
    --model_size_info 1 400                     \
    --dct_coefficient_count 10                  \
    --window_size_ms 40                         \
    --window_stride_ms 20                       \
    --checkpoint=$ckpt                          \
    --output_file=train/$model/frozen.pb
}


function lstm {
    model=lstm
    ckpt=`ls -t train/$model/training/best/*.ckpt-*.meta | head -1`
    ckpt=${ckpt%.*}
    echo $ckpt

    python train.py                             \
    --model_architecture=$model                 \
    --wanted_words=$words                       \
    --model_size_info 188 500                   \
    --dct_coefficient_count 10                  \
    --window_size_ms 40                         \
    --window_stride_ms 20                       \
    --checkpoint=$ckpt                          \
    --output_file=train/$model/frozen.pb
}

gru
