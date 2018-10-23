from keras.models import Sequential
from keras.layers import Input, Embedding, Dense, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping, LearningRateScheduler
import keras
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from code.util.base_util import get_logger
from code.util.base_util import timer
from code import base_data_process
from code.util import base_util
import os, random
# Suppress warnings
import warnings
import math

ID = 'user_id'
LABEL = 'current_service'
log = get_logger()
TAG = 'test'

category_list = ['gender', 'service_type', 'is_mix_service', 'contract_type',
                 'net_service', 'complaint_level', 'age_group']


def data_prepare():
    df_train, df_test = base_data_process.eda(age2group=True, one_hot=False)
    base_data_process.label2index(df_train, LABEL)
    label = df_train[LABEL]
    df_train.drop(columns=[LABEL], inplace=True)
    df_train.drop(columns=[ID], inplace=True)
    label_one_hot = pd.get_dummies(label)
    feats = [f for f in df_train.columns if f not in category_list]
    log.info('feats are {}'.format(feats))
    category_encode_size_map = {}
    for c in category_list:
        le = preprocessing.LabelEncoder()
        le.fit(pd.concat([df_train[c], df_test[c]], axis=0))

        df_train[c] = le.transform(df_train[c])
        df_test[c] = le.transform(df_test[c])

        category_encode_size_map[c] = len(le.classes_)

        log.info('{} has {} classes, origin classes are {}'.format(c, len(le.classes_), le.classes_))
    return df_train, df_test, label, label_one_hot, feats, category_encode_size_map


def build_model_input_output(feats, category_encode_size_map, l1_size=300, l2_size=100):
    input_feats = Input(shape=(len(feats),), name='number_input')
    input_list = [input_feats]
    embedding_result_list = []
    for c in category_list:
        input_category_c = Input(shape=(1,), name=c)
        # 每次只进去一个数据，获取对应数据的embedding结果，所以input size 为1
        embedding_result = Embedding(output_dim=20, input_dim=category_encode_size_map[c], input_length=1)(
            input_category_c)
        input_list.append(input_category_c)
        embedding_result_list.append(embedding_result)

    category_embeddings_3d = keras.layers.concatenate(embedding_result_list)
    # embedding 出来的是3D张量，带着input length的，因为我们input length 就只有一个，所以需要去掉那一维，这里使用 Flatten 应该也是ok的
    category_embeddings = Reshape((20 * len(category_list),))(category_embeddings_3d)
    dense_layer_in = keras.layers.concatenate([input_feats, category_embeddings])

    l1 = Dense(l1_size, activation='relu')(dense_layer_in)
    l2 = Dense(l2_size, activation='relu')(l1)

    output = Dense(len(base_data_process.decode_list), activation=keras.activations.softmax, name='main_output')(l2)
    return input_list, output


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    df_train, df_test, label, label_one_hot, feats, category_encode_size_map = data_prepare()

    input_list, output = build_model_input_output(feats, category_encode_size_map)

    model = Model(inputs=input_list, outputs=output)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.0002))

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    def get_lr(epoch_num):
        lr = 0.0003 / (1 + epoch_num / 3)
        log.info('epoch:{} lr:{}'.format(epoch_num, lr))
        return lr

    lrs = LearningRateScheduler(get_lr, verbose=1)

    sKF = StratifiedKFold(10, shuffle=True)

    y_pre = np.zeros(shape=(len(df_test), label_one_hot.shape[1]))
    print(label_one_hot.shape)
    i_f = 0
    for train_index, val_index in sKF.split(df_train, label):
        i_f += 1
        log.info('begin fold {}, size of train {}, size of test {}'.format(i_f, len(train_index), len(val_index)))
        train = df_train.loc[train_index]
        val = df_train.loc[val_index]
        data_list = [train[feats]]
        data_val_list = [val[feats]]
        test_data_list = [df_test[feats]]
        for c in category_list:
            data_list.append(train[c])
            data_val_list.append(val[c])
            test_data_list.append(df_test[c])

        model.fit(data_list, label_one_hot.loc[train_index], 30, 100,
                  validation_data=(data_val_list, label_one_hot.loc[val_index]), callbacks=[early_stopping, lrs])
        y_p = model.predict(test_data_list)
        y_pre += y_p
        model.save('keras_model_{}_{}'.format(TAG, i_f))
        base_data_process.write_result('keras_{}_{}.csv'.format(TAG, i_f), df_test[ID], y_p, 'one_hot')

    base_data_process.write_result('keras_{}.csv'.format(TAG), df_test[ID], y_pre, 'one_hot')


def get_layer_out():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model = keras.models.load_model('keras_model_4')
    df_train, df_test, label, label_one_hot, feats, category_encode_size_map = data_prepare()
    test_data_list = [df_test[feats]]
    for c in category_list:
        test_data_list.append(df_test[c])

    from keras import backend as K

    get_layer_out = K.function(
        [model.layers[15].input, model.layers[0].input, model.layers[1].input, model.layers[2].input,
         model.layers[3].input, model.layers[4].input, model.layers[5].input, model.layers[6].input],
        # [model.layers[15].input, model.layers[7].input, model.layers[8].input, model.layers[9].input,
        #  model.layers[10].input, model.layers[11].input, model.layers[12].input, model.layers[13].input],
        [model.layers[-2].output, model.layers[-3].output])

    for i, l in enumerate(model.layers):
        print(i, l.name)
        print(l.input)
        print(l.output)

    print(get_layer_out(test_data_list))


if __name__ == '__main__':
    main()
    # get_layer_out()
