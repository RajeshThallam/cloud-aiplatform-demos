
import argparse
import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _serialize_example(example, label):
  example_value = tf.io.serialize_tensor(example).numpy()
  label_value = tf.io.serialize_tensor(label).numpy()
  feature = {
      'examples':
          tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[example_value])),
      'labels':
          tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_value])),
  }
  return tf.train.Example(features=tf.train.Features(
      feature=feature)).SerializeToString()


def _tf_serialize_example(example, label):
  serialized_tensor = tf.py_function(_serialize_example, (example, label),
                                     tf.string)
  return tf.reshape(serialized_tensor, ())


def generate_examples(training_data_uri, test_data_uri, config_file_uri):
  (train_data, test_data), info = tfds.load(
      # Use the version pre-encoded with an ~8k vocabulary.
      'imdb_reviews/subwords8k',
      # Return the train/test datasets as a tuple.
      split=(tfds.Split.TRAIN, tfds.Split.TEST),
      # Return (example, label) pairs from the dataset (instead of a dictionary).
      as_supervised=True,
      with_info=True)

  serialized_train_examples = train_data.map(_tf_serialize_example)
  serialized_test_examples = test_data.map(_tf_serialize_example)

  filename = os.path.join(training_data_uri, "train.tfrecord")
  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(serialized_train_examples)

  filename = os.path.join(test_data_uri, "test.tfrecord")
  writer = tf.data.experimental.TFRecordWriter(filename)
  writer.write(serialized_test_examples)

  encoder = info.features['text'].encoder
  config = {
      'vocab_size': encoder.vocab_size,
  }
  config_file = os.path.join(config_file_uri, "config")
  with tf.io.gfile.GFile(config_file, 'w') as f:
    f.write(json.dumps(config))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_data_uri', type=str)
  parser.add_argument('--test_data_uri', type=str)
  parser.add_argument('--config_file_uri', type=str)

  args = parser.parse_args()
  generate_examples(args.training_data_uri, args.test_data_uri,
                    args.config_file_uri)
