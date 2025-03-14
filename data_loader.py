# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os

import numpy as np
import tensorflow as tf

import jax.numpy as jnp
from core.data import tf_io

_int64_feature = tf_io.int64_feature
_float_feature = tf_io.float_feature
_bytes_feature = tf_io.bytes_feature
_int64_scalar_feature = tf_io.int64_scalar_feature
_int64_sequence_feature = tf_io.int64_sequence_feature
_string_scalar_feature = tf_io.string_scalar_feature




def decode_fn(record_bytes, include_strings=False):
  features = {
      'tokens': _int64_sequence_feature(),
      'docstring_tokens': _int64_sequence_feature(),
      'edge_sources': _int64_sequence_feature(),
      'edge_dests': _int64_sequence_feature(),
      'edge_types': _int64_sequence_feature(),
      'node_token_span_starts': _int64_sequence_feature(),
      'node_token_span_ends': _int64_sequence_feature(),
      'token_node_indexes': _int64_sequence_feature(),
      'true_branch_nodes': _int64_sequence_feature(),
      'false_branch_nodes': _int64_sequence_feature(),
      'raise_nodes': _int64_sequence_feature(),
      'start_index': _int64_scalar_feature(),
      'exit_index': _int64_scalar_feature(),
      'step_limit': _int64_scalar_feature(),
      'target': _int64_scalar_feature(),
      'target_lineno': _int64_scalar_feature(),
      'target_node_indexes': _int64_sequence_feature(),
      'num_target_nodes': _int64_scalar_feature(),
      'post_domination_matrix': _int64_sequence_feature(),
      'post_domination_matrix_shape': _int64_sequence_feature(),

      'in_dataset': _int64_scalar_feature(),
      'num_tokens': _int64_scalar_feature(),
      'num_nodes': _int64_scalar_feature(),
      'num_edges': _int64_scalar_feature(),
  }
  if include_strings:
    features.update({
        'problem_id': _string_scalar_feature(),
        'submission_id': _string_scalar_feature(),
    })
  example = tf.io.parse_single_example(record_bytes, features)
  example['post_domination_matrix'] = tf.reshape(
      example['post_domination_matrix'],
      example['post_domination_matrix_shape']
  )
  example['edge_sources_shape'] = tf.shape(example['edge_sources'])
  return example



def get_padded_shapes(max_tokens, max_num_nodes, max_num_edges, include_strings=False):
  # We do not expect an error to occur on a line containing more than
  # max_target_nodes statements. Most lines have only a single statement.
  max_target_nodes = 20
  shapes = {
      'tokens': [max_tokens],
      'docstring_tokens': [max_tokens],
      'edge_sources': [2 * max_num_edges + 6],
      'edge_dests': [2 * max_num_edges + 6],
      'edge_types': [2 * max_num_edges + 6],
      'edge_sources_shape': [1],  # Added in trainer.py.
      'node_token_span_starts': [max_num_nodes],
      'node_token_span_ends': [max_num_nodes],
      'token_node_indexes': [max_tokens],
      'true_branch_nodes': [max_num_nodes],
      'false_branch_nodes': [max_num_nodes],
      'raise_nodes': [max_num_nodes],
      'start_index': [1],
      'exit_index': [1],
      'step_limit': [1],
      'target': [1],
      'target_lineno': [1],
      'target_node_indexes': [max_target_nodes],
      'num_target_nodes': [1],
      'post_domination_matrix': [max_num_nodes, max_num_nodes],
      'post_domination_matrix_shape': [2],

      'in_dataset': [1],
      'num_tokens': [1],
      'num_nodes': [1],
      'num_edges': [1],
  }
  if include_strings:
    shapes.update({
        'problem_id': [1],
        'submission_id': [1],
    })
    
  return shapes

def load_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(decode_fn)
    return dataset