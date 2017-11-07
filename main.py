#!/usr/bin/env python3

import random
import pandas as pd
import numpy as np
import time
import logging
import argparse
import lightgbm as lgb

from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from numba import jit

SEED = 265359275

FOLDS = 5


def log_duration(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        logging.info(
                'Function %s took %fs', func.__name__, time.time() - start)
        return ret
    return wrapper


def merge_dicts(d1, d2):
    for k, v in d2.items():
        if k not in d1:
            d1[k] = v
        else:
            d1[k] += v


class LGBMFactory:
    def __init__(self, **args):
        self.args = args

    def __call__(self):
        return LGBM(**self.args)


class LGBM:
    def __init__(
            self, num_rounds=100, eta=0.3, min_child_weight=1,
            subsample=0.7, colsample=0.7, num_leaves=31,
            lambda_l1=0, lambda_l2=0, scale_pos_weight=1,
            min_split_gain=1):
        self.num_rounds = num_rounds
        self.eta = eta
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample = colsample
        self.num_leaves = num_leaves
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.scale_pos_weight = scale_pos_weight
        self.min_split_gain = min_split_gain

    def _MakeArgs(self, train_set, evals=None, early_stopping_rounds=None):
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': self.eta,
            'feature_fraction': self.colsample,
            'bagging_fraction': self.subsample,
            'bagging_freq': 1,
            'num_leaves': self.num_leaves,
            'verbose': 0,
            'seed': SEED,
            'lambda_l1': self.lambda_l1,
            'lambda_l2': self.lambda_l2,
            'min_child_weight': self.min_child_weight,
            'scale_pos_weight': self.scale_pos_weight,
            'min_split_gain': self.min_split_gain,
        }

        args = dict(
            params=params,
            train_set=train_set,
            num_boost_round=self.num_rounds,
            verbose_eval=50,
            feval=gini_xgb,
        )

        if evals:
            args['valid_sets'] = evals
        if early_stopping_rounds is not None:
            args['early_stopping_rounds'] = early_stopping_rounds

        return args

    def predict(self, X):
        return self.gbm.predict(X, num_iteration=self.gbm.best_iteration)

    @log_duration
    def fit(self, train_X, train_y, test_X, test_y):
        lgb_train = lgb.Dataset(train_X, train_y)
        evals = None
        if test_y is not None:
            evals = lgb.Dataset(test_X, test_y)

        # specify your configurations as a dict
        args = self._MakeArgs(
                lgb_train, evals=evals,
                early_stopping_rounds=50)
        # train
        self.gbm = lgb.train(**args)

        self.imp = dict(zip(self.gbm.feature_name(), self.gbm.feature_importance()))


@jit
def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = eval_gini(labels, preds)
    return ('gini', gini_score, True)


def CostFunction(y_true, y_pred):
    return eval_gini(y_true, y_pred)


def CrossValidate(train, test, target, variables, model_factory):
    loss = []
    importance = {}

    select = KFold(FOLDS, shuffle=True, random_state=SEED)
    test_preds = np.zeros(len(test))
    train_preds = np.zeros(len(train))
    sub_test = test
    for train_index, test_index in select.split(train, train[target]):
        sub_train = train.iloc[train_index].copy()
        sub_val = train.iloc[test_index].copy()

        sub_features = []
        sub_train, sub_test, sub_val, sub_features = gen_specialized_features(
                sub_train, test, sub_val, target)
        sub_features = variables + sub_features

        model = model_factory()
        model.fit(
                sub_train[sub_features], sub_train[target],
                sub_val[sub_features], sub_val[target])

        merge_dicts(importance, model.imp)

        val_pred = model.predict(sub_val[sub_features])
        cv = CostFunction(np.array(sub_val[target]), np.array(val_pred))
        loss.append(cv)
        train_preds[test_index] = val_pred

        test_pred = model.predict(sub_test[sub_features])
        test_preds += test_pred

        logging.info('Fold CV: %f, Running CV mean: %f', cv, np.mean(loss))
    test_preds /= FOLDS
    return np.mean(loss), test_preds, importance


@log_duration
def gen_features(df):
    categorical = list(name for name in df.columns if name.endswith('_cat'))
    for cat in categorical:
        df[cat] += 1
    return df


@log_duration
def gen_mean_feature_for_target(column_name, train, test, val, target):
    df = pd.DataFrame(
            {column_name: train[column_name], 'target': train[target]})
    grouped = df.groupby(column_name)['target'].agg(['count', 'mean'])
    grouped = grouped[grouped['count'] > 200]

    output_name = '{}_mean_{}'.format(column_name, target)

    tmp = train.join(grouped, on=column_name).fillna(-1)
    train.loc[:, output_name] = tmp['mean']

    tmp = test.join(grouped, on=column_name).fillna(-1)
    test.loc[:, output_name] = tmp['mean']

    tmp = val.join(grouped, on=column_name).fillna(-1)
    val.loc[:, output_name] = tmp['mean']

    return train, test, val, output_name


def compute_hcc(
        train_df, test_df, val_df, variable,
        target, min_samples_leaf=1, smoothing=1):
    prior_prob = train_df[target].mean()

    hcc_name = "_".join(['hcc', variable, target])

    grouped = train_df.groupby(variable)[target].agg(["size", "mean"])

    smoothing = 1/(1+np.exp(-(grouped["size"]-min_samples_leaf)/smoothing))

    grouped[hcc_name] = (
            prior_prob * (1 - smoothing) +
            grouped["mean"] * smoothing)

    grouped = grouped[[hcc_name]]

    assert variable in train_df.columns
    assert variable in test_df.columns
    assert variable in val_df.columns

    train_df = train_df.join(grouped, on=variable)
    test_df = test_df.join(grouped, on=variable)
    val_df = val_df.join(grouped, on=variable)

    return train_df, test_df, val_df, hcc_name


@log_duration
def hcc_encode(train_df, test_df, val_df, variable, target):
    """
    See "A Preprocessing Scheme for High-Cardinality Categorical Attributes in
    Classification and Prediction Problems" by Daniele Micci-Barreca
    """
    return compute_hcc(
            train_df, test_df, val_df, variable, target,
            min_samples_leaf=200, smoothing=10)


@log_duration
def binarize_feature(train, test, column):
    data = train[column].append(test[column])

    grouped = data.groupby(data).size()

    grouped = grouped[grouped >= 5]

    f = grouped.to_dict()

    tmp = data.apply(lambda x: f.get(x, None) and x)
    d = pd.get_dummies(tmp, prefix=column)

    train = train.join(d)
    test = test.join(d)

    feat_names = d.columns.tolist()

    return train, test, feat_names


@log_duration
def single_out(train_df, test_df, variable):
    data = train_df[variable].append(test_df[variable])
    grouped = data.groupby(data).agg(['size'])

    train_df.loc[
            train_df.join(grouped, on=variable)['size'] <= 1, variable] = -1
    test_df.loc[
            test_df.join(grouped, on=variable)['size'] <= 1, variable] = -1

    return train_df, test_df


@log_duration
def gen_specialized_features(train, test, val, target):
    features = []

    categorical = list(name for name in train.columns if name.endswith('_cat'))
    for cat in categorical:
        train, test, val, feature = hcc_encode(
                train, test, val, cat, target)
        features.append(feature)

    return train, test, val, features


@log_duration
def preCVFeatures(train, test):
    combs = [
        ('ps_reg_01', 'ps_car_02_cat'),
        ('ps_reg_01', 'ps_car_04_cat'),
    ]

    features = []

    for n_c, (f1, f2) in enumerate(combs):
        name1 = f1 + "_plus_" + f2 + '_joined'
        print('current feature %60s %4d' % (name1, n_c + 1))

        train[name1] = (
                train[f1].apply(lambda x: str(x)) + "_" +
                train[f2].apply(lambda x: str(x)))
        test[name1] = (
                test[f1].apply(lambda x: str(x)) + "_" +
                test[f2].apply(lambda x: str(x)))
        # Label Encode
        lbl = LabelEncoder()
        lbl.fit(list(train[name1].values) + list(test[name1].values))
        train[name1] = lbl.transform(list(train[name1].values))
        test[name1] = lbl.transform(list(test[name1].values))

        features.append(name1)

    return train, test, features


@log_duration
def load_train_data():
    df = pd.read_csv('train.csv.gz')
    print(df.describe())
    return df


@log_duration
def load_test_data():
    df = pd.read_csv('test.csv.gz')
    print(df.describe())
    return df


def product_params_rec(keys, args, acc):
    if not keys:
        yield dict(acc)
        return None
    for v in args[keys[0]]:
        acc[keys[0]] = v
        yield from product_params_rec(keys[1:], args, acc)
        del acc[keys[0]]


def product_params(args):
    keys = sorted(list(args.keys()))
    output = list(product_params_rec(keys, args, {}))
    r = random.SystemRandom()
    r.shuffle(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--rounds', default=20000, type=int,
            help='Max number of boosting rounds')
    parser.add_argument(
            '--eta', default=0.07, type=float, help='Learning rate')
    parser.add_argument(
            '--full', default=False, action='store_true',
            help='Whether to do full train and evaluate test')
    parser.add_argument(
            '--search', default=False, action='store_true',
            help='Whether to run hyperparam search')
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    logging.basicConfig(
            format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
            level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    logging.info('Reading data...')
    train = gen_features(load_train_data())
    test = gen_features(load_test_data())

    feats = [
      # 'ps_ind_01',
      # 'ps_ind_02_cat',
      # 'ps_ind_03',
      # 'ps_ind_04_cat',
      # 'ps_ind_05_cat',
      # 'ps_ind_06_bin',
      # 'ps_ind_07_bin',
      # 'ps_ind_08_bin',
      # 'ps_ind_09_bin',
      # 'ps_ind_10_bin',
      # 'ps_ind_11_bin',
      # 'ps_ind_12_bin',
      # 'ps_ind_13_bin',
      # 'ps_ind_14',
      # 'ps_ind_15',
      # 'ps_ind_16_bin',
      # 'ps_ind_17_bin',
      # 'ps_ind_18_bin',
      # 'ps_reg_01',
      # 'ps_reg_02',
      # 'ps_reg_03',
      # 'ps_car_01_cat',
      # 'ps_car_02_cat',
      # 'ps_car_03_cat',
      # 'ps_car_04_cat',
      # 'ps_car_05_cat',
      # 'ps_car_06_cat',
      # 'ps_car_07_cat',
      # 'ps_car_08_cat',
      # 'ps_car_09_cat',
      # 'ps_car_10_cat',
      # 'ps_car_11_cat',
      # 'ps_car_11',
      # 'ps_car_12',
      # 'ps_car_13',
      # 'ps_car_14',
      # 'ps_car_15',
      # 'ps_calc_01',
      # 'ps_calc_02',
      # 'ps_calc_03',
      # 'ps_calc_04',
      # 'ps_calc_05',
      # 'ps_calc_06',
      # 'ps_calc_07',
      # 'ps_calc_08',
      # 'ps_calc_09',
      # 'ps_calc_10',
      # 'ps_calc_11',
      # 'ps_calc_12',
      # 'ps_calc_13',
      # 'ps_calc_14',
      # 'ps_calc_15_bin',
      # 'ps_calc_16_bin',
      # 'ps_calc_17_bin',
      # 'ps_calc_18_bin',
      # 'ps_calc_19_bin',
      # 'ps_calc_20_bin',

      "ps_car_13",
      "ps_reg_03",
      "ps_ind_05_cat",
      "ps_ind_03",
      "ps_ind_15",
      "ps_reg_02",
      "ps_car_14",
      "ps_car_12",
      "ps_car_01_cat",
      "ps_car_07_cat",
      "ps_ind_17_bin",
      "ps_car_03_cat",
      "ps_reg_01",
      "ps_car_15",
      "ps_ind_01",
      "ps_ind_16_bin",
      "ps_ind_07_bin",
      "ps_car_06_cat",
      "ps_car_04_cat",
      "ps_ind_06_bin",
      "ps_car_09_cat",
      "ps_car_02_cat",
      "ps_ind_02_cat",
      "ps_car_11",
      "ps_car_05_cat",
      "ps_calc_09",
      "ps_calc_05",
      "ps_ind_08_bin",
      "ps_car_08_cat",
      "ps_ind_09_bin",
      "ps_ind_04_cat",
      "ps_ind_18_bin",
      "ps_ind_12_bin",
      "ps_ind_14",
    ]

    TARGET = 'target'
    ID_COLUMN = 'id'

    train = train[feats + [TARGET, ID_COLUMN]]
    test = test[feats + [ID_COLUMN]]

    # train, test, f = preCVFeatures(train, test)
    # feats.extend(f)

    param_dict = dict(
            num_rounds=args.rounds,
            eta=args.eta,
            subsample=0.9,
            colsample=0.8,
            lambda_l1=8,
            lambda_l2=1.8,
            num_leaves=25,
            min_child_weight=6,
            scale_pos_weight=1.6,
            min_split_gain=1)

    if args.search:
        params_space = dict(
            num_leaves=[10, 15, 20, 25, 31],
        )

        best_score = -1e100
        best_params = None

        history = []

        param_combinations = product_params(params_space)
        logging.info(
                'Search %d param combinations...', len(param_combinations))
        for params in param_combinations:
            params = dict(params)
            logging.info('-----------------------------------------------')
            logging.info('Params: %s', params)
            p = dict(param_dict)
            p.update(params)

            factory = LGBMFactory(**p)
            score, _, _ = CrossValidate(
                    train, test, TARGET, feats, model_factory=factory)
            logging.info('Score: %f', score)
            if score > best_score:
                best_score = score
                best_params = params
                logging.info('Best score so far')
            logging.info('Current best score: %f', best_score)
            logging.info('Current best params: %s', best_params)
            history.append((score, params))

        for score, params in sorted(history, key=lambda x: x[0]):
            logging.info('='*80)
            logging.info('Score: %f', score)
            logging.info('Params: %s', params)

        param_dict.update(best_params)

    if args.full:
        factory = LGBMFactory(**param_dict)
        logging.info('Training...')
        loss, test_pred, imp = CrossValidate(
                train, test, TARGET, feats, model_factory=factory)

        for f, w in sorted(imp.items(), key=lambda x: x[1]):
            logging.info('%s %d', f, w)

        logging.info('Prediction mean: %f', test_pred.mean())

        logging.info('Train mean: %f', train[TARGET].mean())

        logging.info('CV Score: %f', loss)

        test[TARGET] = test_pred

        test = test.sort_values(ID_COLUMN)

        test.to_csv(
                'solution.csv.gz',
                columns=[ID_COLUMN] + [TARGET],
                index=False, compression='gzip')


if __name__ == '__main__':
    main()
