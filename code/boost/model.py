import pandas as pd
import time
import numpy as np
import gc
import time
import lightgbm as lgb
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from code.util.base_util import timer
from code.util.base_util import get_logger
from code.util.base_util import pickle_dump
from code.base_data_process import eda, one_hot2label_index, label2index, index2label, write_result

log = get_logger()
ID = 'user_id'
LABEL = 'current_service'

# TAG = 'stacking'
TAG = 'stacking_0.5'


def model(train, test, num_folds=5, stratified=True, num_boost_round=1000, save_path='origin_data_save'):
    LABEL_SIZE = train[LABEL].value_counts().count()

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train.shape, test.shape))

    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    sub_preds = np.zeros(shape=(test.shape[0], LABEL_SIZE))
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train.columns if
             f not in [LABEL, ID]]
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train[LABEL])):
        dtrain = lgb.Dataset(data=train[feats].iloc[train_idx],
                             label=train[LABEL].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train[feats].iloc[valid_idx],
                             label=train[LABEL].iloc[valid_idx],
                             free_raw_data=False, silent=True)

        pickle_dump((train_idx, valid_idx), (save_path + '/index_boost_{}_{}.pkl').format(TAG, i_fold))
        # params = {
        #     'bagging_fraction': 0.94795171020152,
        #     'bagging_freq': 6,
        #     'bin_construct_sample_cnt': 200000,
        #     'boosting_type': 'gbdt',
        #     'feature_fraction': 0.9953235660931046,
        #     'is_unbalance': False,
        #     'learning_rate': 0.01,
        #     'min_data_in_leaf': 30,
        #     'num_class': 11,
        #     'num_leaves': 80,
        #     'num_threads': 40,
        #     'objective': 'multiclass',
        #     'reg_alpha': 0.001,
        #     'reg_lambda': 0.1,
        #     # 'verbose': -1
        # }

        params = {'bagging_fraction': 0.6, 'bagging_freq': 6, 'bin_construct_sample_cnt': 200000,
                  'boosting_type': 'gbdt', 'feature_fraction': 0.5, 'is_unbalance': True,
                  'learning_rate': 0.01,
                  'min_data_in_leaf': 20, 'num_class': 11, 'num_leaves': 400, 'num_threads': 40,
                  'objective': 'multiclass',
                  'reg_alpha': 0, 'reg_lambda': 0, 'verbose': -1}

        # params = {'bagging_fraction': 0.953396263495823, 'bagging_freq': 6, 'bin_construct_sample_cnt': 20000,
        #           'boosting_type': 'gbdt', 'feature_fraction': 0.817768534641304, 'is_unbalance': True,
        #           'learning_rate': 0.01,
        #           'min_data_in_leaf': 20, 'num_class': 11, 'num_leaves': 400, 'num_threads': 40,
        #           'objective': 'multiclass',
        #           'reg_alpha': 0.2, 'reg_lambda': 0, 'verbose': 0}
        with timer('fold {} train model'.format(i_fold)):
            clf = lgb.train(
                num_boost_round=num_boost_round,
                params=params,
                train_set=dtrain,
                valid_sets=[dvalid],
                early_stopping_rounds=50
            )
            clf.save_model((save_path + '/model_{}_{}_{}.txt').format(TAG, i_fold, int(time.time())))
        with timer('fold {} predict'.format(i_fold)):
            v_data = clf.predict(dvalid.data)
            y_pre = one_hot2label_index(v_data)
            sub_preds += clf.predict(test[feats])

            write_result('result_{}_{}.csv'.format(TAG, i_fold), test[ID], sub_preds, 'one_hot')
            # write2file(test[ID], sub_preds, TAG + str(i_fold))
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = i_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        f1 = f1_score(dvalid.label, y_pre, average='macro')
        log.warn('Fold {} f1 : {} score {}'.format(i_fold + 1, f1, f1 ** 2))
        del clf, dtrain, dvalid
        gc.collect()
    display_importances(feature_importance_df, 'lgbm_importances_{}.png'.format(TAG))


# Display/plot feature importance
def display_importances(feature_importance_df_, name):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(name)


if __name__ == '__main__':
    if not os.path.exists('origin_data_save'):
        os.mkdir('origin_data_save')
    with timer('data process'):
        df_train, df_test = eda(age2group=True, one_hot=False, scale=False)
        cols_index_to_use = [59, 90, 98, 9, 33, 79, 63, 7, 44, 19, 47, 74, 38, 66]
        cols_to_use = ['col_{}'.format(i) for i in cols_index_to_use]

        ll_df = pd.read_csv('../../origin_data/last_layer_100.csv', index_col=False, header=0)
        ll_df_test = pd.read_csv('../../origin_data/last_layer_100_test.csv', index_col=False, header=0)

        ll_df = ll_df[cols_to_use]
        ll_df_test = ll_df_test[cols_to_use]

        print('before concat train shape:{},test shape{}'.format(ll_df.shape, ll_df_test.shape))

        df_train_cols = list(df_train.columns)
        df_train_cols.extend(ll_df.columns)
        df_train = pd.concat([df_train, ll_df], axis=1, ignore_index=True)
        df_train.columns = df_train_cols

        df_test_cols = list(df_test.columns)
        df_test_cols.extend(ll_df_test.columns)
        df_test = pd.concat([df_test, ll_df_test], axis=1, ignore_index=True)
        df_test.columns = df_test_cols
        print('train shape:{},test shape{}'.format(df_train.shape, df_test.shape))

        # df_train.corr().to_csv('corr.csv')

        print(df_train.columns)
        label2index(df_train, LABEL)
    with timer('model process'):
        model(df_train, df_test, num_folds=5, num_boost_round=10000)
