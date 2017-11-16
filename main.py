#!/usr/bin/env python3

import random
import pandas as pd
import numpy as np
import logging
import argparse
import common

# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

ID_COLUMN = 'id'


@common.log_duration
def gen_features(df):
    categorical = list(name for name in df.columns if name.endswith('_cat'))
    for cat in categorical:
        df[cat] += 1
    return df


@common.log_duration
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


@common.log_duration
def single_out(train_df, test_df, variable):
    data = train_df[variable].append(test_df[variable])
    grouped = data.groupby(data).agg(['size'])

    train_df.loc[
            train_df.join(grouped, on=variable)['size'] <= 1, variable] = -1
    test_df.loc[
            test_df.join(grouped, on=variable)['size'] <= 1, variable] = -1

    return train_df, test_df


class SpecializedFeatures:
    def __init__(self, target):
        self.target = target

    def fit(self, train, test=None):
        categorical = list(
                name for name in train.columns if name.endswith('_cat'))
        classifiers = []
        for cat in categorical:
            classifiers.append(common.HCCEncoder(cat, self.target, 200, 10))

        self.classifier = common.NestedClassifiers(classifiers)
        self.classifier.fit(train)

    def predict(self, X):
        return self.classifier.predict(X)


@common.log_duration
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


@common.log_duration
def load_train_data():
    df = pd.read_csv('train.csv.gz', index_col=ID_COLUMN)
    print(df.describe())
    return df


@common.log_duration
def load_test_data():
    df = pd.read_csv('test.csv.gz', index_col=ID_COLUMN)
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


def MakeModelFactory(target, lgbm_params):
    return lambda: common.CrossValidator(
            target,
            common.NestedClassifiersFactory([
                lambda: SpecializedFeatures(target),
                common.UpsamplerFactory(
                    target, common.LGBMFactory(target, **lgbm_params)),
                ]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--rounds', default=650, type=int,
            help='Max number of boosting rounds')
    parser.add_argument(
            '--eta', default=0.02, type=float, help='Learning rate')
    parser.add_argument(
            '--full', default=False, action='store_true',
            help='Whether to do full train and evaluate test')
    parser.add_argument(
            '--search', default=False, action='store_true',
            help='Whether to run hyperparam search')
    args = parser.parse_args()

    random.seed(common.SEED)
    np.random.seed(common.SEED)

    logging.basicConfig(
            format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
            level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    logging.info('Reading data...')
    train = gen_features(load_train_data())
    test = gen_features(load_test_data())

    feats = [
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

    train = train[feats + [TARGET]]
    test = test[feats]

    # train, test, f = preCVFeatures(train, test)
    # feats.extend(f)

    param_dict1 = dict(
            num_rounds=args.rounds,
            eta=args.eta,
            subsample=0.9,
            colsample=0.8,
            lambda_l1=8,
            lambda_l2=1.8,
            num_leaves=25,
            min_child_weight=6,
            scale_pos_weight=1.6,
            min_split_gain=1,
            early_stopping=None,
    )

    # param_dict2 = dict(
    #         num_rounds=args.rounds,
    #         eta=args.eta,
    #         subsample=0.8,
    #         colsample=0.8,
    #         lambda_l1=8,
    #         lambda_l2=3,
    #         num_leaves=15,
    #         min_child_weight=6,
    #         scale_pos_weight=1.6,
    #         min_split_gain=2,
    #         max_depth=4)

    # param_dict3 = dict(
    #         num_rounds=args.rounds,
    #         eta=args.eta,
    #         max_depth=4)

    param_dicts = [param_dict1]

    if args.search:
        params_space = dict(
            num_rounds=[700, 750, 650],
        )

        best_score = -1e100
        best_params = None
        best_model = None

        history = []

        param_combinations = product_params(params_space)
        logging.info(
                'Search %d param combinations...', len(param_combinations))
        for params in param_combinations:
            params = dict(params)
            logging.info('-----------------------------------------------')
            logging.info('Params: %s', params)
            p = dict(param_dicts[0])
            p.update(params)

            factory = MakeModelFactory(TARGET, p)
            validator = factory()
            validator.fit(train)
            score = validator.loss

            logging.info('Score: %f', score)
            if score > best_score:
                best_score = score
                best_params = params
                best_model = validator
                logging.info('Best score so far')
            logging.info('Current best score: %f', best_score)
            logging.info('Current best params: %s', best_params)
            history.append((score, params))

        for score, params in sorted(history, key=lambda x: x[0]):
            logging.info('='*80)
            logging.info('Score: %f', score)
            logging.info('Params: %s', params)

        test_pred = best_model.predict(test)

        logging.info('Prediction mean: %f', test_pred.mean())
        logging.info('Train mean: %f', train[TARGET].mean())

        test[TARGET] = test_pred
        train[TARGET] = validator.train_preds

        common.SaveDF(test, 'solution-lgbm.csv.gz', columns=[TARGET])
        common.SaveDF(train, 'train-lgbm.csv.gz', columns=[TARGET])

    if args.full:
        factory = MakeModelFactory(TARGET, *param_dicts)
        logging.info('Training...')
        validator = factory()
        validator.fit(train)
        logging.info('CV Score: %f', validator.loss)

        logging.info('Evaluating test set...')
        test_pred = validator.predict(test)

        logging.info('Prediction mean: %f', test_pred.mean())
        logging.info('Train mean: %f', train[TARGET].mean())

        test[TARGET] = test_pred
        train[TARGET] = validator.train_preds

        common.SaveDF(test, 'solution-lgbm.csv.gz', columns=[TARGET])
        common.SaveDF(train, 'train-lgbm.csv.gz', columns=[TARGET])


if __name__ == '__main__':
    main()
