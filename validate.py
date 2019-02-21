# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Runs a trained audio graph against WAVE files and reports the results.

The model, labels and .wav files specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav_dir.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav_dir=/tmp/speech_dataset/left

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys

import tensorflow as tf

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import

input_name  = 'wav_data:0'
output_name = 'labels_softmax:0'
num_top_predictions = 3

wav_dir = 'data/validation'

words = [
    "вредно",
    "запрещено",
    "исключено",
    "не_делай",
    "не_надо",
    "не_нужно",
    "не_следует",
    "не_стоит",
    "нельзя",
    "плохая_примета",
    "разве_можно"]

models = [
    "cnn",
    "crnn",
    "dnn",
    "ds_cnn",
    "gru",
    "lstm",
]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  tf.reset_default_graph()
  with tf.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(graph, labels):
    
  """Runs the audio data through the graph and prints predictions."""
  stat = {}
  with tf.Session() as sess:
    for word in words:
        path = f'{wav_dir}/{word}/*.wav'
        # counter, ok, error
        stat[word] = [0,0,0]
     
        for wav_path in glob.glob(path):
            
          if not wav_path or not tf.gfile.Exists(wav_path):
            tf.logging.fatal('Audio file does not exist %s', wav_path)

          with open(wav_path, 'rb') as wav_file:
            wav_data = wav_file.read()

          softmax_tensor = sess.graph.get_tensor_by_name(output_name)
          predictions, = sess.run(softmax_tensor, {input_name: wav_data})

          
          # Sort to show labels in order of confidence
          top_k = predictions.argsort()[-num_top_predictions:][::-1]
          i = top_k[0]
          l = labels[i]
          # print(f'{l} {predictions[i]:.2f}')
          stat[word][0]+=1
          if l=="_unknown_": l="other"
          if l == word: stat[word][1] +=1
          else:         stat[word][2] +=1
          
          # for node_id in top_k:
          #   human_string = labels[node_id]
          #   score = predictions[node_id]
          #   print('%s (score = %.5f)' % (human_string, score))
    tf.logging.warn(stat)
    return stat


def validate_model(model):
  """Loads the model and labels, and runs the inference to print predictions."""

  tf.logging.warn(f'Model: {model}')
  
  labels = f'train/{model}/training/{model}_labels.txt'
  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  graph = f'train/{model}/{model}_frozen.pb'
  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)
  tf.logging.warn(graph)

  labels_list = load_labels(labels)
  # load graph, which is stored in the default session
  load_graph(graph)

  return run_graph(graph, labels_list)

stat = {}
for m in models:
    stat[m] = validate_model(m)


for k, m in stat.items():
    print(k)
    for k, s in m.items():
        print(f'\t{k}\t{s[0]}\t{s[1]/s[0]:.2f}')
