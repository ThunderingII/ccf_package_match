import pandas as pd
import numpy as np

from sklearn import preprocessing
from code.util.base_util import get_logger
import os
import pickle

LABEL = 'current_service'
ID = 'user_id'

log = get_logger()


def base_data_prepare(age2group=True, one_hot=True):
    extra_path = ''
    if not os.getcwd().endswith('311'):
        extra_path = '../'

    df_train = pd.read_csv(extra_path + '../origin_data/train.csv')
    df_test = pd.read_csv(extra_path + '../origin_data/test.csv')

    # None process
    none_list = ['age', '2_total_fee', '3_total_fee', 'gender']
    remove_N_with_Nan(df_train, none_list)
    remove_N_with_Nan(df_test, none_list)

    fill_mean_list = ['2_total_fee', '3_total_fee']
    fill_nan_mean(df_train, fill_mean_list, df_train)
    fill_nan_mean(df_test, fill_mean_list, df_train)

    # process age
    df_train['age'].fillna(df_train['age'].value_counts().index[0], inplace=True)
    df_train['age'] = df_train['age'].astype(np.int32)

    df_test['age'].fillna(df_train['age'].value_counts().index[0], inplace=True)
    df_test['age'] = df_test['age'].astype(np.int32)

    # process gender
    def gender_process(x):
        if '1' in str(x):
            return 1
        elif '2' in str(x):
            return 2
        else:
            return 0

    df_train['gender'] = df_train['gender'].apply(gender_process)
    df_test['gender'] = df_test['gender'].apply(gender_process)

    if age2group:
        # process age
        df_train['age_group'] = df_train['age'].apply(lambda x: x // 10)
        df_test['age_group'] = df_test['age'].apply(lambda x: x // 10)

    # one hot
    category_list = ['gender', 'service_type', 'is_mix_service', 'contract_type',
                     'net_service', 'complaint_level', 'age_group']

    if one_hot:
        dummies(df_train, category_list)
        dummies(df_test, category_list)

    label = df_train[LABEL]

    df_train['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']
    df_test['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']

    df_train['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)
    df_test['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)

    print('before align\ntrain shape:{}\ntest shape:{}'.format(df_train.shape, df_test.shape))

    df_train, df_test = df_train.align(df_test, join='inner', axis=1)

    print('after align\ntrain shape:{}\ntest shape:{}'.format(df_train.shape, df_test.shape))

    df_train[LABEL] = label

    return df_train, df_test


def remove_N_with_Nan(df, columns):
    # 替换数据里面的\N缺失
    for c in columns:
        df[c] = df[c].replace('\\N', np.nan)


def fill_nan_mean(df, columns, df_value, as_type=np.float):
    # 用mean去填充nan数据

    for c in columns:
        df[c] = df[c].astype(as_type)
        df[c].fillna(df_value[c].mean(), inplace=True)


def dummies(df, columns):
    le = preprocessing.LabelEncoder()
    if df is None:
        return
    for c in columns:
        if c not in df.columns:
            print('{} not in df'.format(c))
            continue
        if df[c].value_counts().count() <= 2:
            le.fit(df[c])
            df[c] = le.transform(df[c])
        else:
            d = pd.get_dummies(df[c], prefix=c)
            print('{} has divide into:\n\t\t\t\t{}'.format(c, list(d.columns)))
            for nc in d.columns:
                df[nc] = d[nc]
            df.drop(columns=[c], inplace=True)


def one_hot2label_index(y_pre_origin):
    y_pre_np = np.array(y_pre_origin)
    log.info('data shape {},data length {}'.format(y_pre_np.shape, len(y_pre_np)))
    return np.argmax(y_pre_origin, 1)


encode_map = {}
decode_list = []


# get label index from label's value
def label2index(df, label_name, inplace=True):
    global decode_list
    global encode_map
    if not len(decode_list):
        decode_list = sorted(list(df[label_name].unique()))

    label_size = len(decode_list)

    for i in range(label_size):
        encode_map[decode_list[i]] = i
        log.info('{} \'s index is {}'.format(decode_list[i], i))
    t = df[label_name]
    t = t.apply(lambda x: encode_map[x])
    if inplace:
        df[label_name] = t
    print(df[label_name].value_counts())
    log.info('-' * 100)
    return t


def index2label(y_pre_label_index):
    return list(map(lambda x: decode_list[x], y_pre_label_index))


def write_result(file_name, id_series, label_list, label_type='one_hot'):
    # todo 完成数据的写入部分

    '''
    :param file_name:
    :param id_series:
    :param label_list:
    :param label_type:
    :return:
    '''


def eda(age2group=True, one_hot=True):
    train, test = base_data_prepare(age2group, one_hot)
    # todo eda code

    return train, test
