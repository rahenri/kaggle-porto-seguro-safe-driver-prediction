#!/usr/bin/env python3

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

    df = df.sample(frac=1)

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
    base_model = common.AveragerFactory(target, [0.5, params['lgbm_weight']])
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

    model_params = dict(
        lgbm_weight=0.6,
    )

    if args.search:
        params_space = dict(
            lgbm_weight=[
                0.5, 0.6, 0.7, 0.8, 0.9,
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        )

        def ModelFactory(**params):
            p = dict(model_params)
            p.update(params)
            return MakeModelFactory(TARGET, p)()
        model = common.HyperparamSearch(ModelFactory, params_space)
        model.fit(train)

        for score, params in sorted(model.history, key=lambda x: x[0]):
            logging.info('='*80)
            logging.info('Score: %f', score)
            logging.info('Params: %s', params)

    elif args.full:
        factory = MakeModelFactory(TARGET, model_params)
        logging.info('Training...')
        model = factory()
        model.fit(train)
    else:
        print('No action requested')
        return

    logging.info('Score: %f', model.loss)

    logging.info('Evaluating test set...')
    test = gen_features(load_test_data())
    test_pred = model.predict(test)

    logging.info('Prediction mean: %f', test_pred.mean())
    logging.info('Train mean: %f', train[TARGET].mean())

    test[TARGET] = test_pred
    common.SaveDF(test, 'solution-stack.csv.gz', [TARGET])


if __name__ == '__main__':
    main()
