import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from sklearn.model_selection import KFold, StratifiedKFold
from code.util.base_util import timer
import os
from code import base_data_process
import tensorflow as tf
from code.util import base_util

log = base_util.get_logger()

ID_COLUMN_NAME = 'user_id'
LABEL_COLUMN_NAME = 'current_service'


def nn_model(df_train, df_test):
    pass


class FeatureNN():

    def __init__(self, x_train, y_train, x_val, y_val, epoch=10, batch_size=30):

        self.epoch = epoch
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.input_size = self.x_train.shape[1]

        # config
        self.f1_out_size = 100
        self.f2_out_size = 100
        self.class_num = 15
        self.decay = 0.9
        self.lr = 0.0001

        self._init_graph()

    # def keras_model(self):

    def _init_graph(self):

        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'

            # Set graph level random seed
            # tf.set_random_seed(self.random_seed)

            # with tf.name_scope('input_data'):
            self.input_x = tf.placeholder(tf.float32, shape=[None, self.input_size], name='train_data')

            self.labels = tf.placeholder(tf.int32, shape=[None], name='train_labels')

            # Model.
            # with tf.name_scope('model'):

            self.f1_out = self.__add_hidden_fc_layer(self.input_x, self.input_size, self.f1_out_size,
                                                     tf.nn.relu, name='fc1', batch_normalization=False)
            self.f2_out = self.__add_hidden_fc_layer(self.f1_out, self.f1_out_size, self.f2_out_size, tf.nn.relu,
                                                     name='fc2', batch_normalization=False)
            self.y_pre = self.__add_hidden_fc_layer(self.f2_out, self.f2_out_size, self.class_num, None, False,
                                                    name='fc_out', batch_normalization=False)

            # bias1 = tf.Variable(tf.constant(value=0.01, shape=[100]), name='{}_b'.format('fc1'))
            # # w = tf.Variable(self.__he_normal([in_dim, out_dim]), name='{}_w'.format(name))
            # w1 = tf.get_variable(name='{}_w'.format('fc1'), shape=[self.input_size, 100],
            #                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # wx_plus_b1 = tf.matmul(self.input_x, w1) + bias1
            # h1 = tf.nn.relu(wx_plus_b1)
            #
            # bias2 = tf.Variable(tf.constant(value=0.01, shape=[100]), name='{}_b'.format('fc2'))
            # # w = tf.Variable(self.__he_normal([in_dim, out_dim]), name='{}_w'.format(name))
            # w2 = tf.get_variable(name='{}_w'.format('fc2'), shape=[100, 100],
            #                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # wx_plus_b2 = tf.matmul(h1, w2) + bias2
            # h2 = tf.nn.relu(wx_plus_b2)
            #
            # bias3 = tf.Variable(tf.constant(value=0.01, shape=[15]), name='{}_b'.format('out'))
            # # w = tf.Variable(self.__he_normal([in_dim, out_dim]), name='{}_w'.format(name))
            # w3 = tf.get_variable(name='{}_w'.format('out'), shape=[100, 15],
            #                      initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
            # self.y_pre = tf.matmul(h2, w3) + bias3

            self.y_softmax = tf.nn.softmax(self.y_pre)
            # self.loss = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.y_softmax), 1))
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.y_pre, name='loss'))

            # Optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='optimizer').minimize(
                self.loss)
            # init
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def __he_normal(self, shape):
        return tf.random_normal(shape, stddev=tf.sqrt(2 / shape[0]))

    def __bn_layer(self, Wx_plus_b):
        fc_mean, fc_var = tf.nn.moments(
            Wx_plus_b,
            axes=[0],  # 想要 normalize 的维度, [0] 代表 batch 维度
            # 如果是图像数据, 可以传入 [0, 1, 2], 相当于求[batch, height, width] 的均值/方差, 注意不要加入 channel 维度
        )
        scale = tf.Variable(tf.ones([Wx_plus_b.shape[1]]))
        shift = tf.Variable(tf.zeros([Wx_plus_b.shape[1]]))
        epsilon = 0.001

        # 修改后:
        ema = tf.train.ExponentialMovingAverage(decay=0.5)  # exponential moving average 的 decay 度

        def mean_var_with_update():
            ema_apply_op = ema.apply([fc_mean, fc_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(fc_mean), tf.identity(fc_var)

        mean, var = mean_var_with_update()  # 根据新的 batch 数据, 记录并稍微修改之前的 mean/var

        Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, mean, var, shift, scale, epsilon)
        return Wx_plus_b

    def __add_hidden_fc_layer(self, fc_in, in_dim, out_dim, activate_func=None, has_bias=True,
                              name='basic hidden fc layer', batch_normalization=True):
        if has_bias:
            bias = tf.Variable(tf.constant(value=0.01, shape=[out_dim]), name='{}_b'.format(name))
        else:
            bias = tf.Variable(tf.zeros([out_dim], name='{}_b'.format(name)))
        w = tf.Variable(self.__he_normal([in_dim, out_dim]), name='{}_w'.format(name))
        # w = tf.get_variable(name='{}_w'.format(name),
        #                     shape=[in_dim, out_dim],
        #                     initializer=tf.contrib.layers.xavier_initializer(),
        #                     dtype=tf.float32)
        wx_plus_b = tf.matmul(fc_in, w) + bias
        if batch_normalization:
            wx_plus_b = self.__bn_layer(wx_plus_b)
        if activate_func is not None:
            return activate_func(wx_plus_b)
        return wx_plus_b

    def train(self):

        log.info('train_begin')

        for epoch in range(self.epoch):
            total_batch = int(len(self.x_train) / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                # generate a batch data
                start_index = i * self.batch_size
                end_index = start_index + self.batch_size

                feed_dict = {self.input_x: self.x_train.iloc[start_index:end_index],
                             self.labels: self.y_train.iloc[start_index:end_index]}
                loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                # if (i + 1) % 100 == 0:
                #     self.lr *= self.decay

                if (i + 1) % 1000 == 0:
                    self.lr *= self.decay
                    log.info('epoch:{} batch:{} finished! loss:{} lr:{}'.format(epoch, i + 1, loss, self.lr))

            y_softmax = self.sess.run(self.y_softmax, feed_dict={self.input_x: self.x_val})

            log.info('y_val shape {},y_pre shape {}'.format(self.y_val.shape, y_softmax.shape))
            print(self.x_val[:20])
            print(self.y_val[:20])
            log.info('-' * 100)
            print(y_softmax[:20])
            print(data_process.one_hot2label_index(y_softmax)[:20])

            # link prediction test
            f1 = f1_score(self.y_val,
                          data_process.one_hot2label_index(y_softmax), average=None)
            log.info(
                'Epoch:{:2d} f1 :{} avg_f1:{:6.4} score:{:6.4f}'.format(epoch + 1, f1, np.mean(f1), np.mean(f1) ** 2))

        log.info('train finished')

        writer = tf.summary.FileWriter('model/graph')
        writer.add_graph(self.sess.graph)
        log.info('to graph finished')

        saver = tf.train.Saver()
        saver.save(self.sess, 'model/sne/model')
        log.info('to model finished')


if __name__ == '__main__':

    df_train, df_test = data_process.base_data_prepare()

    df_label = data_process.label2index(df_train, LABEL_COLUMN_NAME)

    print(df_train.iloc[:20, :])

    df_train.drop([ID_COLUMN_NAME, LABEL_COLUMN_NAME], axis=1, inplace=True)
    df_test.drop([ID_COLUMN_NAME], axis=1, inplace=True)

    log.info('train shape:{}'.format(df_train.shape))
    log.info('label shape:{}'.format(df_label.shape))

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(df_train, df_label)):
        x_train = df_train.iloc[train_idx, :]
        x_val = df_train.iloc[valid_idx, :]
        y_train = df_label.iloc[train_idx]
        y_val = df_label.iloc[valid_idx]

        # model = tf.keras.models.Sequential([
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(200, activation=tf.nn.relu),
        #     tf.keras.layers.Dense(400, activation=tf.nn.relu),
        #     tf.keras.layers.Dropout(0.2),
        #     tf.keras.layers.Dense(15, activation=tf.nn.softmax)
        # ])
        # model.compile(optimizer='adam',
        #               loss='sparse_categorical_crossentropy',
        #               metrics=['accuracy'])
        #
        # model.fit(x_train.values, y_train.values, epochs=5)
        # model.evaluate(x_val.values, y_val.values)

        fn = FeatureNN(x_train, y_train, x_val, y_val)
        fn.train()
