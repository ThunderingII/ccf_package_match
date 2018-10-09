import numpy as np
import os, time, sys
import tensorflow as tf
from code.multi_dnn.nn_prepare_data import batch_yield
from code.base_data_process import write_result
from code.util import base_util
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score


class mul_dnn(object):
    """docstring for mul_dnn"""

    def __init__(self, args, num_tags, input_size, paths, config):
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.optimier = args.optimizer
        self.hidden_dim1 = args.hidden_dim1
        self.hidden_dim2 = args.hidden_dim2
        self.hidden_dim3 = args.hidden_dim3
        self.dropout_keep_prob = args.dropout
        self.beta = args.beta
        self.lr = args.lr
        self.clip_grad = args.clip
        self.optimizer = args.optimizer
        self.test_data_path = args.test_data
        self.num_tags = num_tags
        self.input_size = input_size
        self.config = config
        self.model_path = paths['model_path']
        self.summary_path = paths['summary_path']
        self.logger = base_util.get_logger(paths['log_path'])
        self.result_path = paths['result_path']
        self.gs = 0

    def build_graph(self):
        self.add_placeholders()
        self.mul_dnn_op()
        self.softmax_pred_op()
        self.loss_op()
        self.trainstep_op()
        self.init_op()

    def mul_dnn_op(self):
        with tf.variable_scope('multi_dnn'):
            self.f1_out, W1 = self.hiden_fc_layer(self.features, self.input_size, self.hidden_dim1,
                                                  activate_func=tf.nn.relu, name='fc1')
            self.f2_out, W2 = self.hiden_fc_layer(self.f1_out, self.hidden_dim1, self.hidden_dim2,
                                                  activate_func=tf.nn.relu, name='fc2')
            self.f3_out, W3 = self.hiden_fc_layer(self.f2_out, self.hidden_dim2, self.hidden_dim3,
                                                  activate_func=tf.nn.relu, name='fc3')
            self.logits, W_out = self.hiden_fc_layer(self.f3_out, self.hidden_dim3, self.num_tags, activate_func=None,
                                                     name='out')

            # L2 normalization
            self.regularization = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W_out)

    def hiden_fc_layer(self, input_x, input_size, output_size, activate_func=None, name='basic hidden fc layer'):
        W = tf.get_variable(name='{}_w'.format(name),
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float32)
        b = tf.get_variable(name='{}_b'.format(name),
                            shape=[output_size],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float32)
        h_layer = tf.matmul(input_x, W) + b
        if activate_func is not None:
            return activate_func(h_layer), W
        return h_layer, W

    def add_placeholders(self):
        self.features = tf.placeholder(tf.float32, shape=[None, self.input_size], name='features')
        self.labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        self.dropout_pl = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        self.lr_pl = tf.placeholder(dtype=tf.float32, shape=[], name="lr")

    def softmax_pred_op(self):
        self.pred_label = tf.argmax(self.logits, axis=-1)
        self.pred_label = tf.cast(self.pred_label, tf.int32)

    def loss_op(self):
        # 损失函数
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                labels=self.labels) + self.beta * self.regularization
        self.loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', self.loss)

    def trainstep_op(self):
        with tf.variable_scope('train_step'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            if self.optimizer == 'Adam':
                optim = tf.train.AdamOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adadelta':
                optim = tf.train.AdadeltaOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Adagrad':
                optim = tf.train.AdagradOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'RMSProp':
                optim = tf.train.RMSPropOptimizer(learning_rate=self.lr_pl)
            elif self.optimizer == 'Momentum':
                optim = tf.train.MomentumOptimizer(learning_rate=self.lr_pl, momentum=0.9)
            elif self.optimizer == 'SGD':
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            else:
                optim = tf.train.GradientDescentOptimizer(learning_rate=self.lr_pl)
            grads_and_vars = optim.compute_gradients(self.loss)
            grads_and_vars_clip = [[tf.clip_by_value(g, -self.clip_grad, self.clip_grad), v] for g, v in grads_and_vars]
            self.train_op = optim.apply_gradients(grads_and_vars_clip, global_step=self.global_step)

    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def add_summary(self, sess):
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_path, sess.graph)

    def train(self, train, dev):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=self.config) as sess:
            sess.run(self.init_op)
            self.add_summary(sess)

            for epoch in range(self.epoch_num):
                print('=====epoch====', epoch + 1)
                self.run_one_epoch(sess, train, dev, epoch, saver)

    def test(self, ids, feats_test):
        print('============test=========')
        saver = tf.train.Saver()
        with tf.Session(config=self.config) as sess:
            saver.restore(sess, self.model_path)
            label_pre = self.predict(sess, feats_test)
            submit_path = os.path.join(self.result_path, 'result.csv')
            write_result(submit_path, ids, label_pre)

    def run_one_epoch(self, sess, train, dev, epoch, saver):
        num_batches = (len(train) + self.batch_size - 1) // self.batch_size
        batches = batch_yield(train, self.batch_size)
        for step, (feats, label) in enumerate(batches):
            self.global_step += epoch * num_batches + step + 1
            feed_dict = self.get_feed_dict(feats, label, self.lr, self.dropout_keep_prob)
            _, loss_train, summary, step_num_, test_labels, test_logists = sess.run(
                [self.train_op, self.loss, self.merged, self.global_step, self.labels, self.logits],
                feed_dict=feed_dict)
            if step + 1 == 1 or (step + 1) % 100 == 0 or step + 1 == num_batches:
                self.logger.info('epoch {}, step {},loss: {:.4},global_step: {}'.format(epoch + 1, step + 1,
                                                                                        loss_train, step_num_))
            self.file_writer.add_summary(summary, step_num_)

        saver.save(sess, self.model_path, global_step=sess.run(self.global_step))

        self.logger.info('===========validation / test============')
        # because batch size is -1,the 'for' will run exactly 1 time
        for feats, label_true in batch_yield(dev, -1):
            label_pre = self.predict(sess, feats)
            label_true = label_true
            accuracy, precision, recall, f1 = self.evaluate(label_pre, label_true, epoch)
            self.logger.info(
                'accuracy:{},precision:{},recall:{},f1:{}'.format(
                    accuracy, precision, recall, f1
                )
            )

    def get_feed_dict(self, feats, label=None, lr=None, dropout=None):
        feed_dict = {self.features: feats}
        if label is not None:
            # label = []
            # for label_ in label:
            #	label.append(label_)
            feed_dict[self.labels] = label
        if lr is not None:
            feed_dict[self.lr_pl] = lr
        if dropout is not None:
            feed_dict[self.dropout_pl] = dropout
        return feed_dict

    def predict(self, sess, feats):
        feed_dict = self.get_feed_dict(feats, dropout=1.0)
        # print('====feed_dict====',feed_dict)
        label_pre = sess.run(self.pred_label, feed_dict=feed_dict)
        return label_pre

    def evaluate(self, label_pre, label_true, epoch=None):
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        try:
            accuracy = accuracy_score(label_true, label_pre)
            precision = precision_score(label_true, label_pre, average='micro')
            recall = recall_score(label_true, label_pre, average='micro')
            f1 = f1_score(label_true, label_pre, average='micro')
        except ValueError:
            print('===label_true===', label_true)
            print('===label_pre===', label_pre)
        return accuracy, precision, recall, f1
