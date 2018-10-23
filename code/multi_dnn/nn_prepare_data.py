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

warnings.filterwarnings('ignore')
ID = 'user_id'
LABEL = 'current_service'
log = get_logger()


def data_prepare(df_train, df_test):
    conti_list = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'contract_time',
                  'former_complaint_fee', 'former_complaint_num', 'last_month_traffic', 'local_caller_time',
                  'local_trafffic_month', 'month_traffic', 'online_time', 'pay_num', 'pay_times',
                  'service1_caller_time', 'service2_caller_time', 'pay_num_per_time', 'll']

    normalize_process(df_train, df_test, conti_list)
    # label 2 index
    base_data_process.label2index(df_train, LABEL)

    log.info('current path: {}'.format(os.getcwd()))
    with timer('save train data'):
        df_train.to_csv('../../origin_data/train_modified.csv', index=False)
    with timer('save test data'):
        df_test.to_csv('../../origin_data/test_modified.csv', index=False)


def normalize_process(df_train, df_test, conti_list):
    for col in conti_list:
        df_train[col] = (df_train[col] - df_train[col].min()) / (df_train[col].max() - df_train[col].min())
        df_test[col] = (df_test[col] - df_test[col].min()) / (df_test[col].max() - df_test[col].min())


def read_corpus(corpus_path, shuffle=True):
    df = pd.read_csv(corpus_path)
    if shuffle:
        df.sample(frac=1, replace=True)
    # drop user_id
    id_col = df[ID]
    df.drop([ID], axis=1, inplace=True)
    return id_col, df


def batch_yield(df, batch_size):
    if batch_size == -1:
        batch_size = len(df)
    else:
        # equal to shuffle
        df = df.sample(frac=1)
    total_batch = int(math.ceil(len(df) / batch_size))

    for i in range(total_batch):
        data = df.iloc[i * batch_size:i * batch_size + batch_size, :]
        labels = data[LABEL]
        data.drop([LABEL], axis=1, inplace=True)
        yield data, labels


if __name__ == '__main__':
    df_train, df_test = base_data_process.eda(age2group=True, one_hot=True)

    data_prepare(df_train, df_test)
