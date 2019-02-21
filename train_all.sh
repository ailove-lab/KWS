#!/bin/bash

data_url=https://vs43.ailove.ru:4000/train.tar.gz
words='вредно,нельзя,исключено,не_нужно,не_стоит,разве_можно,запрещено,не_делай,не_надо,не_следует,плохая_примета'
data_dir=data/train
train_dir=train
steps=500,500,500
rate=0.0005,0.0001,0.00002
unknown_percentage=80
dct_coefficient_count=10
window_size_ms=40  
window_stride_ms=20
clip_duration_ms=500
clip_stride_ms=20
sample_rate=16000

function freeze {
    
    # get latest weights
    ckpt=`ls -t $train_dir/$1/training/best/*.ckpt-*.meta | head -1`
    ckpt=${ckpt%.*}
    echo $ckpt

    python freeze.py                               \
    --model_architecture $1                        \
    --model_size_info $2                           \
    --wanted_words=$words                          \
    --dct_coefficient_count=$dct_coefficient_count \
    --clip_duration_ms=$clip_duration_ms           \
    --clip_stride_ms=$clip_stride_ms               \
    --window_size_ms=$window_size_ms               \
    --window_stride_ms=$window_stride_ms           \
    --sample_rate=$sample_rate                     \
    --checkpoint=$ckpt                             \
    --output_file=$train_dir/$1/frozen.pb
}

function train {
    out=$train_dir/$1
    python train.py                                \
    --model_architecture $1                        \
    --model_size_info $2                           \
    --wanted_words=$words                          \
    --data_dir=$data_dir                           \
    --data_url=$data_url                           \
    --unknown_percentage=$unknown_percentage       \
    --dct_coefficient_count=$dct_coefficient_count \
    --clip_duration_ms=$clip_duration_ms           \
    --clip_stride_ms=$clip_stride_ms               \
    --window_size_ms=$window_size_ms               \
    --window_stride_ms=$window_stride_ms           \
    --sample_rate=$sample_rate                     \
    --learning_rate=$rate                          \
    --how_many_training_steps=$steps               \
    --summaries_dir=$train_dir/$1/retrain_logs     \
    --train_dir=$train_dir/$1/training
    
    freeze $1 "$2"
}

train ds_cnn "6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1"
train lstm "188 500"
train crnn "100 10 4 2 1 2 136 188"
train gru "1 400"
train cnn "60 10 4 1 1 76 10 4 2 1 58 128"
train dnn "436 436 436"
