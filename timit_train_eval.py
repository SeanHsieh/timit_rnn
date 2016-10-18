import cPickle as pickle
import os
import random
import time

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer('task_id', 1, 'Classify: 0=dialect, 1=sex, 2=speaker ID, 3=sentence ID')
tf.app.flags.DEFINE_bool('balance_train_set', False, '')
tf.app.flags.DEFINE_float('learning_rate', 0.3, '')
tf.app.flags.DEFINE_integer('batch_size', 4, '')
tf.app.flags.DEFINE_integer('num_epochs', 10, '')
tf.app.flags.DEFINE_integer('size', 100, '')
tf.app.flags.DEFINE_integer('num_layers', 1, '')
tf.app.flags.DEFINE_integer('batches_per_ckpt', 256, '')
tf.app.flags.DEFINE_string('train_pkl_fp', '', '')
tf.app.flags.DEFINE_string('eval_pkl_fp', '', '')
tf.app.flags.DEFINE_string('train_dir', '', '')

FLAGS = tf.app.flags.FLAGS
dtype = tf.float32

def load_timit_data(pkl_fp, task_id=1):
  with open(pkl_fp, 'rb') as f:
    data = pickle.load(f)

  seqs = []
  seq_lens = []
  label_counts = {}
  for metadata, seq in data:
    label = metadata[task_id]
    if label not in label_counts:
      label_counts[label] = 0
    label_counts[label] += 1
    seqs.append((seq, label))
  
  return seqs, label_counts

def prepare_batch(data, label_to_id, max_seq_len, start_from=None):
  batch_x = []
  batch_x_len = []
  batch_y = []

  for _ in xrange(FLAGS.batch_size):
    if start_from == None:
      idx = random.randint(0, len(data) - 1)
    else:
      idx = start_from
      start_from += 1
    seq, target = data[idx]
    seq_len = len(seq)
    target_id = label_to_id[target]

    padding = [np.zeros_like(seq[0])] * (max_seq_len - seq_len)
    seq = seq + padding

    batch_x.append(seq)
    batch_x_len.append(seq_len)
    batch_y.append(target_id)

  return np.array(batch_x), np.array(batch_x_len), np.array(batch_y)

def train():
  # Load data
  train_data, train_label_counts = load_timit_data(FLAGS.train_pkl_fp, FLAGS.task_id)
  eval_data, eval_label_counts = None, {}
  if FLAGS.eval_pkl_fp:
    eval_data, eval_label_counts = load_timit_data(FLAGS.eval_pkl_fp, FLAGS.task_id)

  # Process data
  feat_dim = len(train_data[0][0][0])
  seq_lens = np.array([len(x[0]) for x in train_data])
  max_seq_len = np.max(seq_lens)
  labels = train_label_counts.keys()
  num_classes = len(labels)
  num_labels_min = min(train_label_counts.values())
  print 'Train data: num utterances {}\nlabel counts (n={}), {}\nfeat dim {}\nseq len min/max/mean/std {}/{}/{:.2f}/{:.2f}'.format(len(train_data), num_classes, train_label_counts, feat_dim, np.min(seq_lens), max_seq_len, np.mean(seq_lens), np.std(seq_lens))
  label_to_id = {label : i for i, label in enumerate(labels)}

  # Balance data
  if FLAGS.balance_train_set:
    train_data_balanced = []
    for label in labels:
      train_data_for_label = filter(lambda x: x[1] == label, train_data)
      train_data_balanced += random.sample(train_data_for_label, num_labels_min)
    train_data = train_data_balanced
    print 'Balanced train data: num utterances {}'.format(len(train_data))

  if eval_data:
    num_eval_batches = len(eval_data) // FLAGS.batch_size
    print 'Eval data: num utterances {}\nlabel_counts (n={}), {}'.format(len(eval_data), num_classes, eval_label_counts)

  # Input tensors
  input_seq = tf.placeholder(dtype, shape=[None, max_seq_len, feat_dim], name='input_seq')
  input_seq_len = tf.placeholder(tf.int64, shape=[None], name='input_seq_len')
  label = tf.placeholder(tf.int64, [None], name='target')
  target = tf.one_hot(label, depth=num_classes, dtype=dtype)

  # RNN
  cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.size)
  if FLAGS.num_layers > 1:
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_layers)
  output, state = tf.nn.dynamic_rnn(cell, input_seq, sequence_length=input_seq_len, dtype=dtype)
  state_final = state[-1].h
  
  # Regression
  softmax_w = tf.get_variable('softmax_w', [FLAGS.size, num_classes], dtype=dtype)
  softmax_b = tf.get_variable('softmax_b', [num_classes], dtype=dtype)
  logits = tf.matmul(state_final, softmax_w) + softmax_b
  posterior = tf.nn.softmax(logits)
  # https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html
  cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(posterior), reduction_indices=[1]))
  cross_entropy_summary = tf.scalar_summary('train_cross_entropy_batch', cross_entropy)

  # Updates
  opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
  updates = opt.minimize(cross_entropy)

  # Evaluate
  model_prediction = tf.argmax(posterior, 1)
  correct_prediction = tf.equal(model_prediction, label)
  model_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  model_accuracy_summary = tf.scalar_summary('train_accuracy_batch', model_accuracy)

  # Per-checkpoint summaries (not per batch)
  train_accuracy = tf.placeholder(tf.float32, shape=[], name='train_accuracy')
  train_accuracy_summary = tf.scalar_summary('train_accuracy_ckpt', train_accuracy)
  eval_accuracy = tf.placeholder(tf.float32, shape=[], name='eval_accuracy')
  eval_accuracy_summary = tf.scalar_summary('eval_accuracy_ckpt', eval_accuracy)

  # Initialize tensorflow
  var_init = tf.initialize_all_variables()
  train_summary = tf.merge_summary([cross_entropy_summary, model_accuracy_summary])

  with tf.Session() as sess:
    # Initialize session
    sess.run(var_init)
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # Run training
    batches_per_epoch = len(train_data) // FLAGS.batch_size
    batch_num = 0
    train_accuracy_cumulative = 0.0
    for _ in xrange(FLAGS.num_epochs * batches_per_epoch):
      # Run training
      batch_x, batch_x_len, batch_y = prepare_batch(train_data, label_to_id, max_seq_len)
      train_summary_batch, train_accuracy_batch, _ = sess.run([train_summary, model_accuracy, updates], {input_seq: batch_x, input_seq_len: batch_x_len, label: batch_y})

      summary_writer.add_summary(train_summary_batch, batch_num)
      train_accuracy_cumulative += train_accuracy_batch
      batch_num += 1

      # Run evaluation
      if batch_num % FLAGS.batches_per_ckpt == 0:
        # Train accuracy summary
        train_accuracy_summary_result = sess.run(train_accuracy_summary, feed_dict={train_accuracy: train_accuracy_cumulative / FLAGS.batches_per_ckpt})
        summary_writer.add_summary(train_accuracy_summary_result, batch_num)
        train_accuracy_cumulative = 0.0

        # Eval accuracy summary (full pass through eval data)
        if eval_data:
          eval_accuracy_cumulative = 0.0
          for i in xrange(num_eval_batches):
            # Run evaluation
            batch_x, batch_x_len, batch_y = prepare_batch(eval_data, label_to_id, max_seq_len, start_from=i * FLAGS.batch_size)
            eval_accuracy_batch = sess.run(model_accuracy, {input_seq: batch_x, input_seq_len: batch_x_len, label: batch_y})
            eval_accuracy_cumulative += eval_accuracy_batch
          eval_accuracy_summary_result = sess.run(eval_accuracy_summary, feed_dict={eval_accuracy: eval_accuracy_cumulative / num_eval_batches})
          summary_writer.add_summary(eval_accuracy_summary_result, batch_num)

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
