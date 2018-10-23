import pandas as pd
import numpy as np

from sklearn import preprocessing
from code.util.base_util import get_logger
from code.util.base_util import pickle_load
from code.util.base_util import pickle_dump
from code.util.base_util import timer
import os
import pickle

LABEL = 'current_service'
ID = 'user_id'

log = get_logger()

category_list = ['gender', 'service_type', 'is_mix_service', 'contract_type',
                 'net_service', 'complaint_level', 'age_group', 'ctims']


def __get_dir(dir):
    if not os.getcwd().endswith('code'):
        return '../' + dir
    return dir


def base_data_prepare(age2group=True, one_hot=True):
    df_train = pd.read_csv(__get_dir('../origin_data/train.csv'))
    df_test = pd.read_csv(__get_dir('../origin_data/test.csv'))

    log.info('load data success!\ttrain shape:{}\ttest shape:{}'.format(df_train.shape, df_test.shape))

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

    if one_hot:
        dummies(df_train, category_list)
        dummies(df_test, category_list)

    label = df_train[LABEL]

    df_train['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']
    df_test['pay_num_per_time'] = df_train['pay_num'] / df_train['pay_times']

    df_train['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)
    df_test['ll'] = df_train['last_month_traffic'] / (df_train['local_trafffic_month'] + 1)

    log.info('before align\ttrain shape:{}\ttest shape:{}'.format(df_train.shape, df_test.shape))

    df_train, df_test = df_train.align(df_test, join='inner', axis=1)

    log.info('after align\ttrain shape:{}\ttest shape:{}'.format(df_train.shape, df_test.shape))

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
    if df is None:
        return
    for c in columns:
        if c not in df.columns:
            print('{} not in df'.format(c))
            continue
        else:
            d = pd.get_dummies(df[c], prefix=c)
            print('{} has divide into:\n\t\t\t\t{}'.format(c, list(d.columns)))
            for nc in d.columns:
                df[nc] = d[nc]
            df.drop(columns=[c], inplace=True)


def load_label2index():
    global decode_list
    global encode_map
    if not decode_list:
        encode_map, decode_list = pickle_load(__get_dir('../origin_data/label2index.pkl'))
    # 返回了 map 和list
    return encode_map, decode_list


def one_hot2label_index(y_pre_origin):
    y_pre_np = np.array(y_pre_origin)
    log.info('data shape {},data length {}'.format(y_pre_np.shape, len(y_pre_np)))
    return np.argmax(y_pre_np, 1)


encode_map = {}
decode_list = []


# get label index from label's value
def label2index(df, label_name, inplace=True):
    global decode_list
    global encode_map
    if not len(decode_list):
        decode_list = sorted(list(df[label_name].unique()))

    log.info('find {} classes'.format(len(decode_list)))

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
    pickle_dump((encode_map, decode_list), __get_dir('../origin_data/label2index.pkl'))
    return t


def index2label(y_pre_label_index):
    return list(map(lambda x: decode_list[x], y_pre_label_index))


def write_result(file_name, ids, labels, label_type='label_index'):
    # todo 完成数据的写入部分

    '''
    :param file_name:
    :param id_series:
    :param label_list:
    :param label_type:
    :return:
    '''
    load_label2index()
    df_test = pd.DataFrame()
    df_test[ID] = ids

    if label_type == 'one_hot':
        labels = one_hot2label_index(labels)
    if label_type in ['label_index', 'one_hot']:
        labels = [decode_list[label] for label in labels]
    df_test[LABEL] = labels
    df_test.columns = [ID, 'predict']
    print('====shape df_test====', df_test.shape)
    with timer('write result to {}'.format(file_name)):
        df_test.to_csv(file_name, index=False)


def min_max_scale(df, columns):
    for c in columns:
        df[c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())


def eda(age2group=True, one_hot=True, scale=False):
    pd.option_context('display.width', 2000)

    train, test = base_data_prepare(age2group, one_hot)
    # ['service_type', 'is_mix_service', 'online_time', '1_total_fee',
    #  '2_total_fee', '3_total_fee', '4_total_fee', 'month_traffic',
    #  'many_over_bill', 'contract_type', 'contract_time',
    #  'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num',
    #  'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
    #  'service1_caller_time', 'service2_caller_time', 'gender', 'age',
    #  'complaint_level', 'former_complaint_num', 'former_complaint_fee',
    #  'user_id', 'age_group', 'pay_num_per_time', 'll', 'current_service']
    # print(train.head(5))
    #
    fee_list = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']

    labels = train[LABEL]
    train.drop(columns=[LABEL], inplace=True)
    train['is_train'] = True
    test['is_train'] = False

    df = pd.concat((train, test))
    print(df.shape)

    df['fee_max'] = df[fee_list].max(0)

    df['fee_min'] = df[fee_list].min(0)

    df['fee_mean'] = df[fee_list].mean(0)

    df['fee_std'] = df[fee_list].std(0)

    df['mt_m_lmt'] = df['month_traffic'] - df['last_month_traffic']

    df['ctims'] = df['contract_type'].map(str) + '_' + df['is_mix_service'].map(str)

    continue_list = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'contract_time',
                     'former_complaint_fee', 'former_complaint_num', 'last_month_traffic', 'local_caller_time',
                     'local_trafffic_month', 'month_traffic', 'online_time', 'pay_num', 'pay_times',
                     'service1_caller_time', 'service2_caller_time', 'pay_num_per_time', 'll', 'fee_max', 'fee_min',
                     'fee_mean', 'fee_std', 'mt_m_lmt']

    if scale:
        min_max_scale(df, continue_list)
    for c in category_list:
        if c in df.columns:
            df[c], _ = pd.factorize(df[c], sort=True)
        else:
            print('==>', c, 'not in the columns')

    print(df.columns)
    print(df.shape)

    df_train = df[df['is_train'] == True]
    df_test = df[df['is_train'] == False]

    df_train[LABEL] = labels
    print(df_train.head(5))

    return df_train, df_test
    # return train, test


if __name__ == '__main__':
    eda(one_hot=False)
