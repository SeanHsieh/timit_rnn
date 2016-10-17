import cPickle as pickle
import os
import random
import time

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_bool("balance_train_set", False, "")
tf.app.flags.DEFINE_float("learning_rate", 0.3, "")
tf.app.flags.DEFINE_integer("batch_size", 4, "")
tf.app.flags.DEFINE_integer("num_epochs", 10, "")
tf.app.flags.DEFINE_integer("size", 100, "")
tf.app.flags.DEFINE_integer("num_layers", 1, "")
tf.app.flags.DEFINE_integer("batches_per_ckpt", 256, "")
tf.app.flags.DEFINE_string("train_pkl_fp", "", "")
tf.app.flags.DEFINE_string("eval_pkl_fp", "", "")
tf.app.flags.DEFINE_string("train_dir", "", "")

FLAGS = tf.app.flags.FLAGS
dtype = tf.float32

def load_timit_data(pkl_fp):
  with open(pkl_fp, 'rb') as f:
    data = pickle.load(f)

  seqs = []
  seq_lens = []
  label_counts = {}
  for metadata, seq in data:
    dialect, sex, speaker_id, sentence_id = metadata
    label = sex
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
  train_data, train_label_counts = load_timit_data(FLAGS.train_pkl_fp)
  eval_data, eval_label_counts = None, {}
  if FLAGS.eval_pkl_fp:
    eval_data, eval_label_counts = load_timit_data(FLAGS.eval_pkl_fp)

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
    print 'Eval data: num utterances {}\nlabel_counts (n={}), {}'.format(len(eval_data), num_classes, eval_label_counts)

  with tf.Session() as sess:
    # Input
    input_seq = tf.placeholder(dtype, shape=[None, max_seq_len, feat_dim], name="input_seq")
    input_seq_len = tf.placeholder(tf.int32, shape=[None], name="input_seq_len")
    label = tf.placeholder(tf.int32, [None], name="target")

    # RNN
    cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.size)
    if FLAGS.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_layers)
    output, _ = tf.nn.dynamic_rnn(cell, input_seq, sequence_length=input_seq_len, dtype=dtype)
    output_T = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output_T, int(output_T.get_shape()[0]) - 1)
    
    # Regression
    softmax_w = tf.get_variable("softmax_w", [FLAGS.size, num_classes], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=dtype)
    logits = tf.matmul(last, softmax_w) + softmax_b
    prediction = tf.nn.softmax(logits)
    target = tf.one_hot(label, depth=num_classes, dtype=dtype)
    # https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(target * tf.log(prediction), reduction_indices=[1]))
    cross_entropy_summary = tf.scalar_summary("cross_entropy", cross_entropy)

    # Updates
    opt = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    updates = opt.minimize(cross_entropy)

    # Evaluate
    correct_predictions = tf.equal(tf.cast(tf.argmax(prediction, 1), np.int32), label)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype))

    # Extra summaries
    eval_accuracy = tf.placeholder(tf.float32, shape=[], name="eval_accuracy")
    eval_accuracy_summary = tf.scalar_summary("eval_accuracy", eval_accuracy)

    # Initialize tensorflow
    sess.run(tf.initialize_all_variables())
    train_summary = tf.merge_summary([cross_entropy_summary])
    eval_summary = tf.merge_summary([eval_accuracy_summary])
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    # Run training
    batches = 0
    for epoch in xrange(FLAGS.num_epochs):
      for _ in xrange(len(train_data) // FLAGS.batch_size):
        batch_x, batch_x_len, batch_y = prepare_batch(train_data, label_to_id, max_seq_len)

        # Run training
        batch_train_summary, batch_loss, _ = sess.run([train_summary, cross_entropy, updates], {input_seq: batch_x, input_seq_len: batch_x_len, label: batch_y})
        summary_writer.add_summary(batch_train_summary, batches)

        # Run evaluation
        if batches % FLAGS.batches_per_ckpt == 0:
          if eval_data:
            num_batches = len(eval_data) // FLAGS.batch_size
            accuracy_cumulative = 0.0
            for i in xrange(num_batches):
              batch_x, batch_x_len, batch_y = prepare_batch(eval_data, label_to_id, max_seq_len, start_from=i * FLAGS.batch_size)

              # Run evaluation
              batch_accuracy = sess.run(accuracy, {input_seq: batch_x, input_seq_len: batch_x_len, label: batch_y})
              accuracy_cumulative += batch_accuracy
            eval_summary_result = sess.run(eval_summary, feed_dict={eval_accuracy: accuracy_cumulative / num_batches})
            summary_writer.add_summary(eval_summary_result, batches)

        batches += 1

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
