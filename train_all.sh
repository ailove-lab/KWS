#!/bin/bash

# data_url=https://vs43.ailove.ru:4000/train.tar.gz
data_url=''
# wanted_words='вредно,нельзя,исключено,не_нужно,не_стоит,разве_можно,запрещено,не_делай,не_надо,не_следует,плохая_примета'
wanted_words='вредно,нельзя,исключено,запрещено'
data_dir=data/train
train_dir=train
how_many_training_steps=500,500,500
learning_rate=0.0005,0.0001,0.00002
unknown_percentage=80
dct_coefficient_count=20
window_size_ms=40
window_stride_ms=20
clip_duration_ms=1000
time_shift_ms=400
clip_stride_ms=20
sample_rate=16000
background_volume=0.2 

function freeze {
    # get latest weights
    ckpt=`ls -t $train_dir/$1/training/best/*.ckpt-*.meta | head -1`
    ckpt=${ckpt%.*}
    echo $ckpt
    python freeze.py                               \
    --model_architecture $1                        \
    --model_size_info $2                           \
    --wanted_words=$wanted_words                   \
    --dct_coefficient_count=$dct_coefficient_count \
    --clip_duration_ms=$clip_duration_ms           \
    --clip_stride_ms=$clip_stride_ms               \
    --window_size_ms=$window_size_ms               \
    --window_stride_ms=$window_stride_ms           \
    --sample_rate=$sample_rate                     \
    --checkpoint=$ckpt                             \
    --output_file=$train_dir/$1/${1}_frozen.pb
}

function train {
    out=$train_dir/$1
    rm -rf $out
    python train.py                                    \
    --model_architecture $1                            \
    --model_size_info $2                               \
    --wanted_words=$wanted_words                       \
    --data_dir=$data_dir                               \
    --data_url=$data_url                               \
    --unknown_percentage=$unknown_percentage           \
    --dct_coefficient_count=$dct_coefficient_count     \
    --clip_duration_ms=$clip_duration_ms               \
    --clip_stride_ms=$clip_stride_ms                   \
    --window_size_ms=$window_size_ms                   \
    --window_stride_ms=$window_stride_ms               \
    --time_shift_ms=$time_shift_ms                     \
    --sample_rate=$sample_rate                         \
    --learning_rate=$learning_rate                     \
    --how_many_training_steps=$how_many_training_steps \
    --summaries_dir=$train_dir/$1/retrain_logs         \
    --train_dir=$train_dir/$1/training
    
    freeze $1 "$2"
}

# train ds_cnn "6 276 10 4 2 1 276 3 3 2 2 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1 276 3 3 1 1"
# train crnn "100 10 4 2 1 2 136 188"
# train gru "1 400"
#train lstm "188 500"
#train cnn "60 10 4 1 1 76 10 4 2 1 58 128"
#train dnn "436 436 436"
function test {
    stat=./gru_steps.tsv
    rm $stat
    for i in 100 200 300 500 600
    do
        echo $i
        rm -rf $train_dir
        mkdir -p $train_dir
        # unknown_percentage=$i
        # steps=$i,$i,$i
        train gru "1 400"
        echo Validation
        echo $i >> $stat
        ./build/validate >> $stat
    done
}

train gru "1 400"

# --data_url
#   default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
#   Location of speech training data archive on the web.
# --data_dir
#   default='/tmp/speech_dataset/',
#   Where to download the speech training data to. 
# --background_volume
#   default=0.1,
#   How loud the background noise should be between 0 and 1. 
# --background_frequency
#   default=0.8,
#   How many of the training samples have background noise mixed in. 
# --silence_percentage
#   default=10.0,
#   How much of the training data should be silence. 
# --unknown_percentage
#   default=10.0,
#   How much of the training data should be unknown words. 
# --time_shift_ms
#   default=100.0,
#   Range to randomly shift the training audio by in time. 
# --testing_percentage
#   default=10,
#   What percentage of wavs to use as a test set.
# --validation_percentage
#   default=10,
#   What percentage of wavs to use as a validation set.
# --sample_rate
#   default=16000,
#   Expected sample rate of the wavs
# --clip_duration_ms
#   default=1000,
#   Expected duration in milliseconds of the wavs
# --window_size_ms
#   default=30.0,
#   How long each spectrogram timeslice is
# --window_stride_ms
#   default=10.0,
#   How long each spectrogram timeslice is
# --dct_coefficient_count
#   default=40,
#   How many bins to use for the MFCC fingerprint
# --how_many_training_steps
#   default='15000,3000',
#   How many training loops to run
# --eval_step_interval
#   default=400,
#   How often to evaluate the training results.
# --learning_rate
#   default='0.001,0.0001',
#   How large a learning rate to use when training.
# --batch_size
#   default=100,
#   How many items to train with at once
# --summaries_dir
#   default='/tmp/retrain_logs',
#   Where to save summary logs for TensorBoard.
# --wanted_words
#   default='yes,no,up,down,left,right,on,off,stop,go',
#   Words to use (others will be added to an unknown label
# --train_dir
#   default='/tmp/speech_commands_train',
#   Directory to write event logs and checkpoint.
# --save_step_interval
#   default=100,
#   Save model checkpoint every save_steps.
# --start_checkpoint
#   default='',
#   If specified restore this pretrained model before any training.
# --model_architecture
#   default='dnn',
#   What model architecture to use
# --model_size_info
#   nargs="+",
#   default=[128,128,128],
#   Model dimensions - different for various models
# --check_nans
#   default=False,
#   Whether to check for invalid numbers during processing

