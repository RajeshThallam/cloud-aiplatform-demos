
import argparse
import json
import os

import numpy as np
import tensorflow as tf


def _parse_example(record):
  f = {
      'examples': tf.io.FixedLenFeature((), tf.string, default_value=''),
      'labels': tf.io.FixedLenFeature((), tf.string, default_value='')
  }
  return tf.io.parse_single_example(record, f)


def _to_tensor(record):
  examples = tf.io.parse_tensor(record['examples'], tf.int64)
  labels = tf.io.parse_tensor(record['labels'], tf.int64)
  return (examples, labels)


def train_examples(training_data_uri, test_data_uri, config_file_uri,
                   output_model_uri, output_metrics_uri):
  train_examples = tf.data.TFRecordDataset(
      [os.path.join(training_data_uri, 'train.tfrecord')])
  test_examples = tf.data.TFRecordDataset(
      [os.path.join(test_data_uri, 'test.tfrecord')])

  train_batches = train_examples.map(_parse_example).map(_to_tensor)
  test_batches = test_examples.map(_parse_example).map(_to_tensor)

  with tf.io.gfile.GFile(os.path.join(config_file_uri, 'config')) as f:
    config = json.loads(f.read())

  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(config['vocab_size'], 16),
      tf.keras.layers.GlobalAveragePooling1D(),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  model.summary()

  model.compile(
      optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  train_batches = train_batches.shuffle(1000).padded_batch(
      32, (tf.TensorShape([None]), tf.TensorShape([])))

  test_batches = test_batches.padded_batch(
      32, (tf.TensorShape([None]), tf.TensorShape([])))

  history = model.fit(
      train_batches,
      epochs=10,
      validation_data=test_batches,
      validation_steps=30)

  loss, accuracy = model.evaluate(test_batches)

  metrics = {
      'loss': str(loss),
      'accuracy': str(accuracy),
  }

  model_json = model.to_json()
  with tf.io.gfile.GFile(os.path.join(output_model_uri, 'model.json'),
                         'w') as f:
    f.write(model_json)

  with tf.io.gfile.GFile(os.path.join(output_metrics_uri, 'metrics.json'),
                         'w') as f:
    f.write(json.dumps(metrics))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--training_data_uri', type=str)
  parser.add_argument('--test_data_uri', type=str)
  parser.add_argument('--config_file_uri', type=str)
  parser.add_argument('--output_model_uri', type=str)
  parser.add_argument('--output_metrics_uri', type=str)

  args = parser.parse_args()

  train_examples(args.training_data_uri, args.test_data_uri,
                 args.config_file_uri, args.output_model_uri,
                 args.output_metrics_uri)
