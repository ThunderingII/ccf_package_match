import csv
from hyperopt import STATUS_OK
from code.util.base_util import timer
from code.util.base_util import get_logger
import lightgbm as lgb
import numpy as np
import pickle
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
from sklearn.metrics import f1_score
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from code.base_data_process import eda
from code.base_data_process import label2index
from code.base_data_process import one_hot2label_index

ITERATION = 0

log = get_logger()


def cross_validation(train, params, ID_COLUMN_NAME, LABEL_COLUMN_NAME, N_FOLD=5):
    '''
    :return: loss
    '''
    NUM_BOOST_ROUND = 5
    EARLY_STOPPING_ROUNDS = 2

    # Cross validation model
    folds = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=1001)
    feats = [f for f in train.columns if
             f not in [LABEL_COLUMN_NAME, ID_COLUMN_NAME]]
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train[LABEL_COLUMN_NAME])):
        dtrain = lgb.Dataset(data=train[feats].iloc[train_idx],
                             label=train[LABEL_COLUMN_NAME].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train[feats].iloc[valid_idx],
                             label=train[LABEL_COLUMN_NAME].iloc[valid_idx],
                             free_raw_data=False, silent=True)
        with timer('cross validation-fold {} train model'.format(i_fold)):
            log.info('params is {}'.format(params))
            clf = lgb.train(
                num_boost_round=NUM_BOOST_ROUND,
                params=params,
                verbose_eval=10,
                train_set=dtrain,
                valid_sets=[dvalid],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS
            )
        with timer('cross validation-fold {} predict'.format(i_fold)):
            v_data = clf.predict(dvalid.data)
            y_pre = one_hot2label_index(v_data)
        f1 = f1_score(dvalid.label, y_pre, average='macro')
        return f1


def objective(hyperparameters):
    # Keep track of evals
    global ITERATION

    ITERATION += 1

    # Make sure parameters that need to be integers are integers
    for parameter_name in ['num_leaves', 'bin_construct_sample_cnt', 'bagging_freq', 'min_data_in_leaf']:
        hyperparameters[parameter_name] = int(hyperparameters[parameter_name])

    with timer('run lgb') as ti:
        # Perform n_folds cross validation
        f1 = cross_validation(config_dict['train'], hyperparameters, 'user_id', 'current_service')
        loss = 1 - f1 ** 2

        run_time = ti.get_delay_t0()

    # Write to the csv file ('a' means append)
    of_connection = open('hyperparameters.csv', 'a')
    writer = csv.writer(of_connection)
    writer.writerow([ITERATION, loss, hyperparameters, run_time, 1 - loss, f1])
    of_connection.close()

    log.info('iteration-{} f1:{} loss:{} train_time:{}'.format(ITERATION, f1, loss, run_time))
    # Dictionary with information for evaluation
    return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}


def optimization():
    space = {
        'learning_rate': 0.01,
        'boosting_type': hp.choice('boosting_type', ['gbdt']),
        'num_leaves': hp.choice('num_leaves', [15, 20, 30, 50, 65, 80, 100, 150, 400]),
        'bin_construct_sample_cnt': hp.choice('bin_construct_sample_cnt', [10000, 20000, 60000, 100000, 200000]),
        'min_data_in_leaf': hp.quniform('min_data_in_leaf', 20, 500, 10),
        'reg_alpha': hp.choice('reg_alpha', [0, 0.001, 0.01, 0.1, 0.2]),
        'reg_lambda': hp.choice('reg_lambda', [0, 0.001, 0.01, 0.1, 0.2]),
        'feature_fraction': hp.uniform('feature_fraction', 0.8, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.8, 1.0),
        'bagging_freq': hp.choice('bagging_freq', [0, 2, 6, 10, 16]),
        'is_unbalance': hp.choice('is_unbalance', [True, False]),
        'num_threads': 40,
        'objective': 'multiclass',
        'num_class': 11,
        'verbose': -1
    }

    trials = Trials()

    with timer('optimization'):
        # Run optimization
        best = fmin(fn=objective, space=space, algo=tpe.suggest, trials=trials,
                    max_evals=config_dict['max_evals'])

    print('-' * 100)
    log.warn(best)

    with open('model_trials.pkl', mode='wb') as mt:
        pickle.dump(trials, mt)


config_dict = {
    'train': pd.DataFrame(),
    'max_evals': 1000
}

if __name__ == '__main__':
    df_train, df_test = eda(True, False)

    config_dict['train'] = df_train.iloc[:, :]
    label2index(df_train, 'current_service')
    optimization()
