from data_loader import decode_fn
import tensorflow as tf
from transformers import PreTrainedTokenizerFast
import sys
from tokenizers import Tokenizer

sys.stdout.reconfigure(encoding='utf-8')


feature_description = {
    'tokens': tf.io.VarLenFeature(tf.int64),
    'docstring_tokens': tf.io.VarLenFeature(tf.int64),
    'edge_sources': tf.io.VarLenFeature(tf.int64),
    'edge_dests': tf.io.VarLenFeature(tf.int64),
    'edge_types': tf.io.VarLenFeature(tf.int64),
    'node_token_span_starts': tf.io.VarLenFeature(tf.int64),
    'node_token_span_ends': tf.io.VarLenFeature(tf.int64),
    'token_node_indexes': tf.io.VarLenFeature(tf.int64),
    'true_branch_nodes': tf.io.VarLenFeature(tf.int64),
    'false_branch_nodes': tf.io.VarLenFeature(tf.int64),
    'raise_nodes': tf.io.VarLenFeature(tf.int64),
    'start_index': tf.io.FixedLenFeature([], tf.int64),
    'exit_index': tf.io.FixedLenFeature([], tf.int64),
    'step_limit': tf.io.FixedLenFeature([], tf.int64),
    'target': tf.io.FixedLenFeature([], tf.int64),
    'target_lineno': tf.io.FixedLenFeature([], tf.int64),
    'target_node_indexes': tf.io.VarLenFeature(tf.int64),
    'num_target_nodes': tf.io.FixedLenFeature([], tf.int64),
    'post_domination_matrix': tf.io.VarLenFeature(tf.int64),
    'post_domination_matrix_shape': tf.io.VarLenFeature(tf.int64),
    'in_dataset': tf.io.FixedLenFeature([], tf.int64),
    'num_tokens': tf.io.FixedLenFeature([], tf.int64),
    'num_nodes': tf.io.FixedLenFeature([], tf.int64),
    'num_edges': tf.io.FixedLenFeature([], tf.int64),
    'problem_id': tf.io.FixedLenFeature([], tf.string),
    'submission_id': tf.io.FixedLenFeature([], tf.string),
}

def _parse_function(proto):
    return tf.io.parse_single_example(proto, feature_description)



# Path to your TFRecord file
file_path = 'dataset/test.tfrecord'

# Create a dataset from the TFRecord file
dataset = tf.data.TFRecordDataset(file_path)

# Parse the dataset
parsed_dataset = dataset.map(_parse_function)
count = 0
# count = sum(1 for _ in parsed_dataset)
# print("Total records:", count)


for record in parsed_dataset:
    target_output = record['target'].numpy()
    problem_id = record['problem_id'].numpy().decode('utf-8')
    if target_output != 1 :
        count+=1
        submission_id = record['submission_id'].numpy().decode('utf-8')
        tokens = record['tokens'].values.numpy()
        docstring_tokens = record['docstring_tokens'].values.numpy()
        lineno = record['target_lineno'].numpy()
       
        print(f'Problem ID: {problem_id}')
        print(f'Submission ID: {submission_id}')
        print(f'Raw Token IDs: {tokens}')  # Debugging step
        tokenizer_docstring =  PreTrainedTokenizerFast(tokenizer_file='core/tokenization/train-docstrings-1000000.json')

        decoded_text = tokenizer_docstring.decode(docstring_tokens.tolist(), skip_special_tokens=True)
        decoded_source_code = tokenizer_docstring.decode(tokens.tolist(),skip_special_tokens=True)
        print(f"Problem Description:\n{decoded_text}")
        print(f"Source Code:{decoded_source_code}")
        print(f"Submission Result: {'PASS' if target_output == 1 else 'FAIL',target_output}")
        print("line number: ", lineno)
        if (count == 6):
            break
    



