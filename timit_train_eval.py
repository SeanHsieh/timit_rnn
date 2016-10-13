import cPickle as pickle
import os
import random
import time

import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_float("learning_rate", 0.3, "")
tf.app.flags.DEFINE_integer("batch_size", 4, "")
tf.app.flags.DEFINE_integer("num_epochs", 10, "")
tf.app.flags.DEFINE_integer("size", 100, "")
tf.app.flags.DEFINE_integer("num_layers", 1, "")
tf.app.flags.DEFINE_integer("batches_per_ckpt", 256, "")
tf.app.flags.DEFINE_string("train_pkl_fp", "", "")
tf.app.flags.DEFINE_string("train_dir", "", "")

FLAGS = tf.app.flags.FLAGS
dtype = tf.float32

def train():
  with open(FLAGS.train_pkl_fp, 'rb') as f:
    train_data = pickle.load(f)
  seqs = []
  label_counts = {}
  for metadata, seq in train_data:
    sex, speaker_id = metadata
    label = sex
    if label not in label_counts:
      label_counts[label] = 0
    label_counts[label] += 1
    seqs.append((seq, label))

  feat_dim = len(seqs[0][0][0])
  seq_lens = np.array([len(seq[0]) for seq in seqs])
  max_seq_len = np.max(seq_lens)
  labels = label_counts.keys()
  num_classes = len(labels)
  print 'num utterances {}\nlabel counts {}\nfeat dim {}\nseq len min/max/mean/std {}/{}/{:.2f}/{:.2f}'.format(sum(label_counts.values()), label_counts, feat_dim, np.min(seq_lens), np.max(seq_lens), np.mean(seq_lens), np.std(seq_lens))
  label_to_id = {label : i for i, label in enumerate(labels)}

  with tf.Session() as sess:
    input_seq = tf.placeholder(dtype, shape=[None, max_seq_len, feat_dim], name="input_seq")
    label = tf.placeholder(tf.int32, [None], name="target")
    target = tf.one_hot(label, depth=num_classes)

    cell = tf.nn.rnn_cell.BasicLSTMCell(FLAGS.size)
    if FLAGS.num_layers > 1:
      cell = tf.nn.rnn_cell.MultiRNNCell([cell] * FLAGS.num_layers)
    output, _ = tf.nn.dynamic_rnn(cell, input_seq, dtype=dtype)
    output_T = tf.transpose(output, [1, 0, 2])
    last = tf.gather(output_T, int(output_T.get_shape()[0]) - 1)
    
    softmax_w = tf.get_variable("softmax_w", [FLAGS.size, num_classes], dtype=dtype)
    softmax_b = tf.get_variable("softmax_b", [num_classes], dtype=dtype)
    logits = tf.matmul(last, softmax_w) + softmax_b
    prediction = tf.nn.softmax(logits)
    cross_entropy = -tf.reduce_sum(target * tf.log(prediction))

    opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    updates = opt.minimize(cross_entropy)

    tf.scalar_summary("cross_entropy", cross_entropy)

    sess.run(tf.initialize_all_variables())

    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    step = 0
    for epoch in xrange(FLAGS.num_epochs):
      for _ in xrange(len(seqs) // FLAGS.batch_size):
        batch_x = []
        batch_y = []
        for _ in xrange(FLAGS.batch_size):
          seq, target = random.choice(seqs)
          seq_len = len(seq)
          target_id = label_to_id[target]

          padding = [np.zeros_like(seq[0])] * (max_seq_len - seq_len)
          seq = seq + padding

          batch_x.append(seq)
          batch_y.append(target_id)
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)

        summary, loss, _ = sess.run([merged, cross_entropy, updates], {input_seq: batch_x, label: batch_y})
        summary_writer.add_summary(summary, step)
        step += 1

def main(_):
  train()

if __name__ == '__main__':
  tf.app.run()
