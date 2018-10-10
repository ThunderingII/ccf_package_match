from keras.models import Sequential
from keras.layers import Input, Embedding, Dense, Reshape
from keras.models import Model
from keras.callbacks import EarlyStopping
import keras
import pandas as pd
import numpy as np

from sklearn import preprocessing
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

category_list = ['gender', 'service_type', 'is_mix_service', 'contract_type',
                 'net_service', 'complaint_level', 'age_group']
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    df_train, df_test = base_data_process.eda(age2group=True, one_hot=False)

    base_data_process.label2index(df_train, LABEL)
    label = df_train[LABEL]
    df_train.drop(columns=[LABEL], inplace=True)
    df_train.drop(columns=[ID], inplace=True)

    label_one_hot = pd.get_dummies(label)

    feats = [f for f in df_train.columns if f not in category_list]

    log.info('feats are {}'.format(feats))

    le_map = {}
    le_size_map = {}
    for c in category_list:
        le = preprocessing.LabelEncoder()
        le.fit(pd.concat([df_train[c], df_test[c]], axis=0))

        df_train[c] = le.transform(df_train[c])
        df_test[c] = le.transform(df_test[c])

        le_map[c] = le
        le_size_map[c] = len(le.classes_)

        log.info('{} has {} classes, origin classes are {}'.format(c, len(le.classes_), le.classes_))

    x_n = Input(shape=(len(feats),), name='number_input')

    embed_map = {}
    embed_in_map = {}
    for c in category_list:
        x_c_in = Input(shape=(1,), name=c)
        e_c = Embedding(output_dim=20, input_dim=le_size_map[c], input_length=None)(x_c_in)
        embed_in_map[c] = x_c_in
        embed_map[c] = e_c

    x_list = []
    x_in_list = [x_n]
    for c in category_list:
        x_list.append(embed_map[c])
        x_in_list.append(embed_in_map[c])

    x_1_n_20 = keras.layers.concatenate(x_list)

    # embedding 出来的是3D张量，带着input length的，因为我们input length 就只有一个，所以需要去掉那一维，这里使用 Flatten 应该也是ok的
    x_reshape = Reshape((20 * len(category_list),))(x_1_n_20)

    x = keras.layers.concatenate([x_n, x_reshape])

    l1 = Dense(100, activation='relu')(x)
    l2 = Dense(100, activation='relu')(l1)

    output = Dense(len(base_data_process.decode_list), activation=keras.activations.softmax, name='main_output')(l2)

    model = Model(inputs=x_in_list, outputs=output)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(lr=0.0002))

    input_list = [df_train[feats]]
    test_input_list = [df_test[feats]]
    for c in category_list:
        input_list.append(df_train[c])
        test_input_list.append(df_test[c])

    early_stopping = EarlyStopping(monitor='loss', patience=5, verbose=1)
    model.fit(input_list, label_one_hot, 30, 100, callbacks=[early_stopping])

    y_pre = model.predict(test_input_list)

    base_data_process.write_result('keras_50.csv', df_test[ID], y_pre, 'one_hot')
