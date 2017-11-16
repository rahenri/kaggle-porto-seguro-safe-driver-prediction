#!/usr/bin/env python3

import random
import pandas as pd
# import numpy as np
import logging
import argparse
import common

# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder

ID_COLUMN = 'id'

TARGET = 'target'


@common.log_duration
def gen_features(df):

    return df


@common.log_duration
def load_train_data():
    df1 = pd.read_csv('train.csv.gz', index_col=ID_COLUMN)
    df1 = df1[[TARGET]]

    df2 = pd.read_csv('train-nn.csv.gz', index_col=ID_COLUMN)
    df2 = df2.rename(columns={TARGET: 'target-nn'})

    df3 = pd.read_csv('train-lgbm.csv.gz', index_col=ID_COLUMN)
    df3 = df3.rename(columns={TARGET: 'target-lgbm'})

    df = df1.join(df2).join(df3)

    return df


@common.log_duration
def load_test_data():
    df1 = pd.read_csv('test.csv.gz', index_col=ID_COLUMN)
    df1 = df1[[]]

    df2 = pd.read_csv('solution-nn.csv.gz', index_col=ID_COLUMN)
    df2 = df2.rename(columns={TARGET: 'target-nn'})

    df3 = pd.read_csv('solution-lgbm.csv.gz', index_col=ID_COLUMN)
    df3 = df3.rename(columns={TARGET: 'target-lgbm'})

    df = df1.join(df2).join(df3)

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


def AveragerFactory(target, weights):
    return lambda: Averager(target, weights)


class Averager:
    def __init__(self, target, weights):
        self.target = target
        self.weights = weights

    def fit(self, train, validation):
        self.features = [f for f in train.columns if f != self.target]
        assert len(self.features) == len(self.weights)

    @common.log_duration
    def predict(self, X):
        W = sum(self.weights)
        return sum([X[f]*w for f, w in zip(self.features, self.weights)]) / W


def MakeModelFactory(target, params):
    # gbm_params = dict(
    #         num_rounds=10000,
    #         eta=0.05,
    #         num_leaves=8,
    #         lambda_l1=8,
    #         lambda_l2=2,
    # )
    # base_model = common.LGBMFactory(target, **gbm_params)
    # return common.UpsamplerFactory(
    #         target,
    #         lambda: common.CrossValidator(target, base_model))

    # base_model = common.SKLearnWrapperFactory(
    #         target, lambda: LogisticRegression())
    base_model = AveragerFactory(target, [0.5, params['lgbm_weight']])
    return common.UpsamplerFactory(
            target,
            lambda: common.CrossValidator(target, base_model))


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

    logging.basicConfig(
            format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
            level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    logging.info('Reading data...')
    train = gen_features(load_train_data())
    test = gen_features(load_test_data())

    model_params = dict(
        lgbm_weight=0.9,
    )

    if args.search:
        params_space = dict(
            lgbm_weight=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
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
            p = dict(model_params)
            p.update(params)

            factory = MakeModelFactory(TARGET, p)
            validator = factory()
            #  common.CrossValidator(TARGET, factory)
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
        common.SaveDF(test, 'solution-stack.csv.gz', [TARGET])

    if args.full:
        factory = MakeModelFactory(TARGET, model_params)
        logging.info('Training...')
        validator = factory()
        validator.fit(train)
        logging.info('CV Score: %f', validator.loss)

        logging.info('Evaluating test set...')
        test_pred = validator.predict(test)

        logging.info('Prediction mean: %f', test_pred.mean())
        logging.info('Train mean: %f', train[TARGET].mean())

        test[TARGET] = test_pred
        common.SaveDF(test, 'solution-stack.csv.gz', [TARGET])


if __name__ == '__main__':
    main()
