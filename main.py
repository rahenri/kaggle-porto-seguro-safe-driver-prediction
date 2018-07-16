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

        def ModelFactory(**params):
            params = dict(params)
            p = dict(param_dicts[0])
            p.update(params)
            return MakeModelFactory(TARGET, p)()
        model = common.HyperparamSearch(ModelFactory, params_space)
        model.fit(train)

        for score, params in sorted(model.history, key=lambda x: x[0]):
            logging.info('='*80)
            logging.info('Score: %f', score)
            logging.info('Params: %s', params)

    elif args.full:
        factory = MakeModelFactory(TARGET, *param_dicts)
        logging.info('Training...')
        model = factory()
        model.fit(train)
        logging.info('CV Score: %f', model.loss)
    else:
        print('No action requested')
        return

    logging.info('Evaluating test set...')

    test = gen_features(load_test_data())
    test = test[feats]

    test_pred = model.predict(test)

    logging.info('Prediction mean: %f', test_pred.mean())
    logging.info('Train mean: %f', train[TARGET].mean())

    test[TARGET] = test_pred
    train[TARGET] = model.train_preds

    common.SaveDF(test, 'solution-lgbm.csv.gz', columns=[TARGET])
    common.SaveDF(train, 'train-lgbm.csv.gz', columns=[TARGET])


if __name__ == '__main__':
    main()
