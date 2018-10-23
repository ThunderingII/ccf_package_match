from keras.models import Sequential
from keras.layers import Input, Embedding, Dense, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
import keras
import pandas as pd
import numpy as np
from keras.utils import plot_model

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from code.util.base_util import get_logger
from code.util.base_util import timer
from code import base_data_process
from code.util import base_util
import os, random
from sklearn.metrics import f1_score
# Suppress warnings
import warnings
import math

ID = 'user_id'
LABEL = 'current_service'
log = get_logger()
TAG = 'max_min_eda'
category_list = ['gender', 'service_type', 'is_mix_service', 'contract_type',
                 'net_service', 'complaint_level', 'age_group', 'ctims']

continue_list = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'contract_time',
                 'former_complaint_fee', 'former_complaint_num', 'last_month_traffic', 'local_caller_time',
                 'local_trafffic_month', 'month_traffic', 'online_time', 'pay_num', 'pay_times',
                 'service1_caller_time', 'service2_caller_time', 'pay_num_per_time', 'll']


def data_prepare():
    df_train, df_test = base_data_process.eda(age2group=True, one_hot=False, scale=True)
    base_data_process.label2index(df_train, LABEL)
    label = df_train[LABEL]
    df_train.drop(columns=[LABEL], inplace=True)
    df_train.drop(columns=[ID], inplace=True)
    label_one_hot = pd.get_dummies(label)
    feats = [f for f in df_train.columns if f not in category_list]
    log.info('feats are {}'.format(feats))
    category_encode_size_map = {}
    for c in category_list:
        if c not in df_train.columns:
            log.warn('{} not in df'.format(c))
            continue
        category_encode_size_map[c] = len(df_train[c].unique())

        log.info('{} has {} classes'.format(c, len(df_train[c].unique())))

        # category_encode_size_map = {}
        # for c in category_list:
        #     if c not in df_train.columns:
        #         continue
        #     le = preprocessing.LabelEncoder()
        #     le.fit(pd.concat([df_train[c], df_test[c]], axis=0))
        #
        #     df_train[c] = le.transform(df_train[c])
        #     df_test[c] = le.transform(df_test[c])
        #
        #     category_encode_size_map[c] = len(le.classes_)

        # log.info('{} has {} classes, origin classes are {}'.format(c, len(le.classes_), le.classes_))

    return df_train, df_test, label, label_one_hot, feats, category_encode_size_map


def build_model_input_output(feats, category_encode_size_map, l1_size=400, l2_size=100, embedding_size=20):
    input_feats = Input(shape=(len(feats),), name='number_input')
    input_list = [input_feats]
    embedding_result_list = []
    for c in category_encode_size_map:
        input_category_c = Input(shape=(1,), name=c)
        # 每次只进去一个数据，获取对应数据的embedding结果，所以input size 为1
        embedding_result = Embedding(output_dim=embedding_size, input_dim=category_encode_size_map[c], input_length=1,
                                     name='{}_embed'.format(c))(input_category_c)
        input_list.append(input_category_c)
        embedding_result_list.append(embedding_result)

    category_embeddings_3d = keras.layers.concatenate(embedding_result_list)
    # embedding 出来的是3D张量，带着input length的，因为我们input length 就只有一个，所以需要去掉那一维，这里使用 Flatten 应该也是ok的
    category_embeddings = Reshape((embedding_size * len(category_encode_size_map),))(category_embeddings_3d)
    dense_layer_in = keras.layers.concatenate([input_feats, category_embeddings])



    l1 = Dense(l1_size, activation='relu', name='hidden1')(dense_layer_in)
    l2 = Dense(l2_size, activation='relu', name='hidden2')(l1)
    ll = Dense(l2_size, activation='relu', name='hidden3')(l2)

    output = Dense(len(base_data_process.decode_list), activation=keras.activations.softmax, name='main_output')(ll)
    return input_list, output


def main(fold_num=5):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    df_train, df_test, label, label_one_hot, feats, category_encode_size_map = data_prepare()

    input_list, output = build_model_input_output(feats, category_encode_size_map)

    def get_lr(epoch_num):
        lr = 0.0005 / (1 + epoch_num / 3)
        log.info('epoch:{} lr:{}'.format(epoch_num, lr))
        return lr

    keras.callbacks.TensorBoard(log_dir='./tb_{}'.format(TAG), histogram_freq=0, write_graph=True,
                                write_images=False,
                                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None,
                                update_freq=10000)

    lrs = LearningRateScheduler(get_lr, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    sKF = StratifiedKFold(fold_num, shuffle=True)

    y_pre = np.zeros(shape=(len(df_test), label_one_hot.shape[1]))
    print(label_one_hot.shape)

    class F1_Call(keras.callbacks.Callback):

        def on_train_begin(self, logs=None):
            logs['f1'] = []
            self.input = None

        def pre(self, s):
            if self.input is None:
                for i, a in enumerate(self.validation_data):
                    print(i, a.shape)
                self.input = self.validation_data[:-2]
                self.label = base_data_process.one_hot2label_index(self.validation_data[-2])
            y_pre = self.model.predict(self.input)

            y_index = base_data_process.one_hot2label_index(y_pre)
            f1 = f1_score(self.label, y_index, average='macro')
            log.info('{} f1: {}'.format(s, f1))
            return f1

        def on_epoch_end(self, epoch, logs=None):
            log.info('============{}=============='.format(epoch))
            f1 = self.pre('epoch {}'.format(epoch))
            if 'f1' not in logs:
                logs['f1'] = []
            logs['f1'].append(f1)
            log.info('==========================')

        def on_batch_end(self, batch, logs=None):
            if (batch + 1) % 7000 == 0:
                self.pre('batch {}'.format(batch))

    i_f = 0
    for train_index, val_index in sKF.split(df_train, label):
        i_f += 1
        base_data_process.pickle_dump((train_index, val_index), ('index_dnn_{}_{}.pkl').format(TAG, i_f))
        log.info('begin fold {}, size of train {}, size of test {}'.format(i_f, len(train_index), len(val_index)))
        train = df_train.loc[train_index]
        val = df_train.loc[val_index]
        data_list = [train[feats]]
        data_val_list = [val[feats]]
        test_data_list = [df_test[feats]]
        for c in category_encode_size_map:
            data_list.append(train[c])
            data_val_list.append(val[c])
            test_data_list.append(df_test[c])

        model = Model(inputs=input_list, outputs=output)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.0002))
        label_val = label_one_hot.loc[val_index]
        model.fit(data_list, label_one_hot.loc[train_index], 30, 100,
                  validation_data=(data_val_list, label_val),
                  callbacks=[early_stopping, lrs, F1_Call()])
        y_pre_fold = model.predict(test_data_list)
        y_pre += y_pre_fold
        model.save('keras_model_{}_{}'.format(TAG, i_f))
        base_data_process.write_result('keras_{}_{}.csv'.format(TAG, i_f), df_test[ID], y_pre_fold, 'one_hot')

    base_data_process.write_result('keras_{}.csv'.format(TAG), df_test[ID], y_pre, 'one_hot')


def get_layer_out():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = keras.models.load_model('keras_model_5')
    plot_model(model, show_shapes=True)
    df_train, df_test, label, label_one_hot, feats, category_encode_size_map = data_prepare()
    test_data_list = [df_test[feats]]
    data_list = [df_train[feats]]
    for c in category_list:
        test_data_list.append(np.reshape(df_test[c].values, (len(df_test[c]), 1)))
        data_list.append(np.reshape(df_train[c].values, (len(df_train[c]), 1)))
    for df in test_data_list:
        print(df.shape)

    for i, l in enumerate(model.layers):
        print(i, l.name)
        print(l.input)
        print(l.output)

    from keras import backend as K

    get_layer_out = K.function(
        [model.layers[15].input, model.layers[0].input, model.layers[1].input, model.layers[2].input,
         model.layers[3].input, model.layers[4].input, model.layers[5].input, model.layers[6].input],
        [model.layers[-2].output, model.layers[-3].output])

    a, b = get_layer_out(data_list)
    a_test, b_test = get_layer_out(test_data_list)

    print('-2 layer shape:\n', a.shape)
    print('-3 layer shape:\n', b.shape)

    import pandas as pd
    cols_100 = ['col_{}'.format(i) for i in range(100)]
    a_df = pd.DataFrame(a, columns=cols_100)
    b_df = pd.DataFrame(b, columns=cols_100)
    a_df_test = pd.DataFrame(a_test, columns=cols_100)
    b_df_test = pd.DataFrame(b_test, columns=cols_100)

    drop_cols_last = [c for c in cols_100 if a_df[c].sum == 0]
    print(drop_cols_last)

    drop_cols_s = [c for c in cols_100 if b_df[c].sum == 0]
    print(drop_cols_s)

    a_df.drop(drop_cols_last, axis=1, inplace=True)
    a_df_test.drop(drop_cols_last, axis=1, inplace=True)

    b_df.drop(drop_cols_s, axis=1, inplace=True)
    b_df_test.drop(drop_cols_s, axis=1, inplace=True)

    print('last_shape:{},second shape:{}'.format(a_df.shape, b_df.shape))
    print('test==>last_shape:{},second shape:{}'.format(a_df_test.shape, b_df_test.shape))
    a_df.to_csv('../../origin_data/last_layer_100.csv', index=False, header=True)
    a_df_test.to_csv('../../origin_data/last_layer_100_test.csv', index=False, header=True)
    b_df.to_csv('../../origin_data/last_2_layer_100.csv', index=False, header=True)
    b_df_test.to_csv('../../origin_data/last_2_layer_100_test.csv', index=False, header=True)

    # get_0_rate(a, 'a')
    # get_0_rate(b, 'b')
    #
    # for i, w in enumerate(model.trainable_weights):
    #     print(i, w)
    #
    # get_0_rate(model.get_weights()[7], 'h1')
    # get_0_rate(model.get_weights()[9], 'h2')


def get_0_rate(data, name):
    t = 0
    df = pd.DataFrame(data)
    for i in range(len(df.columns)):
        m = df.iloc[:, i].mean()
        if m == 0:
            t += 1
        print('==>', name, i, m)
    print('*' * 20, '0 rate:', t / len(df.columns), '*' * 20)


if __name__ == '__main__':
    main()
    # get_layer_out()
