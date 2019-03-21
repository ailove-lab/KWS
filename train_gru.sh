#!/bin/bash
words='вредно,нельзя,исключено,запрещено'
data_dir=data/train
model=gru
out=train/$model

python train.py                             \
--model_architecture $model                 \
--wanted_words=$words                       \
--data_dir=$data_dir                        \
--data_url=                                 \
--unknown_percentage=80                     \
--model_size_info 1 400                     \
--dct_coefficient_count 10                  \
--window_size_ms 40                         \
--window_stride_ms 20                       \
--learning_rate 0.0005,0.0001,0.00002       \
--how_many_training_steps 500,500,500       \
--summaries_dir $out/retrain_logs    \
--train_dir $out/training

ckpt=`ls -t $out/training/best/*.ckpt-*.meta | head -1`
ckpt=${ckpt%.*}
echo $ckpt

python freeze.py                            \
--model_architecture $model                 \
--wanted_words=$words                       \
--data_dir=$data_dir                        \
--unknown_percentage=80                     \
--model_size_info 1 400                     \
--dct_coefficient_count 10                  \
--window_size_ms 40                         \
--window_stride_ms 20                       \
--checkpoint=$ckpt                          \
--output_file=$out/${model}_frozen.pb

cp $out/training/${model}_labels.txt $out/
