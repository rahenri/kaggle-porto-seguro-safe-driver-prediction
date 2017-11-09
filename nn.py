#!/usr/bin/env python3

import random
import pandas as pd
import numpy as np
import logging
import argparse
import common

# from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from keras.callbacks import Callback

ID_COLUMN = 'id'


@common.log_duration
def gen_features(df):
    df['negative_vals'] = np.sum((df == -1).values, axis=1)

    to_drop = {
            'ps_car_11_cat', 'ps_ind_14', 'ps_car_11', 'ps_car_14',
            'ps_ind_06_bin', 'ps_ind_09_bin', 'ps_ind_10_bin',
            'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'}

    cols_use = [
            c for c in df.columns if (not c.startswith('ps_calc_'))
            and (c not in to_drop)]
    df = df[cols_use].copy()

    df = df.replace(-1, np.NaN)

    df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']

    df = df.fillna(-1)

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


class SpecializedFeatures:
    def __init__(self, target):
        self.target = target

    def fit(self, train, test=None):
        encoders = {}
        variables = list(train.columns)
        for var in variables:
            uniq = np.unique(train[var])
            if len(uniq) <= 2 or len(uniq) >= 105:
                continue
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoder.fit(train[[var]]+1)
            encoders[var] = encoder
        self.encoders = encoders

    def predict(self, X):
        X = X.copy()
        for name, enc in self.encoders.items():
            out = enc.transform(X[[name]]+1)
            for i in range(len(out[1])):
                X['{}_{}'.format(name, i)] = out[:, i]

        return X


@common.log_duration
def preCVFeatures(train, test):

    return train, test


@common.log_duration
def load_train_data():
    df = pd.read_csv('train.csv.gz', index_col=ID_COLUMN)
    return df


@common.log_duration
def load_test_data():
    df = pd.read_csv('test.csv.gz', index_col=ID_COLUMN)
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


class NNModelFactory:
    def __init__(self, target):
        self.target = target

    def __call__(self):
        return NNModel(self.target)


class GiniCallback(Callback):
    def __init__(self, target, train, val):
        self.target = target
        self.train = train
        self.val = val

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def compute_gini(self, df):
        y_pred_val = self.model.predict(
                np.array(df.drop(self.target, axis=1)),
                batch_size=2048)
        return common.eval_gini(df[self.target], y_pred_val.reshape([-1]))

    def on_epoch_end(self, epoch, logs={}):
        val_gini = self.compute_gini(self.val)
        train_gini = self.compute_gini(self.train)
        print('\ngini-train: {:.5f} gini-val: {:.5f}'.format(
            train_gini, val_gini))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass


class NNModel:
    def __init__(self, target):
        self.target = target

    def fit(self, train, test=None):
        print(train.columns)
        self.model = Sequential()

        positive = train[self.target] == 1
        train = pd.concat([train, train[positive]])

        train_X = train.drop(self.target, axis=1)
        train_y = train[self.target]

        self.model.add(Dense(
            units=35, activation='relu', input_dim=len(train_X.columns)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=1))
        self.model.add(Activation('sigmoid'))

        optimizer = Adam(lr=0.001)

        self.model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer)

        validation_data = None
        if test is not None:
            validation_data = (
                    np.array(test.drop(self.target, axis=1)),
                    np.array(test[self.target]))

        self.model.fit(
                np.array(train_X),
                np.array(train_y),
                batch_size=2048,
                epochs=15,
                validation_data=validation_data,
                callbacks=[GiniCallback(self.target, train, test)])

    def predict(self, X):
        if self.target in X.columns:
            X = X.drop(self.target, axis=1)
        return self.model.predict(np.array(X), batch_size=2048).reshape([-1])


def MakeModelFactory(target, nn_params):
    base_model = common.NestedClassifiersFactory([
        lambda: SpecializedFeatures(target),
        common.AverageClassifierFactory(
            NNModelFactory(target, **nn_params), 1),
    ])
    return lambda: common.CrossValidator(target, base_model)


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

    random.seed(common.SEED)
    np.random.seed(common.SEED)

    logging.basicConfig(
            format='%(asctime)s %(filename)s:%(lineno)d %(message)s',
            level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')

    logging.info('Reading data...')
    train = gen_features(load_train_data())
    test = gen_features(load_test_data())

    TARGET = 'target'

    nn_params = dict()

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
            p = dict(nn_params)
            p.update(params)

            factory = MakeModelFactory(TARGET, p)
            validator = common.CrossValidator(TARGET, factory)
            validator.fit(train)
            score = validator.loss

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

    if args.full:
        factory = MakeModelFactory(TARGET, nn_params)
        logging.info('Training...')
        validator = factory()
        validator.fit(train)
        logging.info('CV Score: %f', validator.loss)

        logging.info('Evaluating test set...')
        test_pred = validator.predict(test)

        logging.info('Prediction mean: %f', test_pred.mean())
        logging.info('Train mean: %f', train[TARGET].mean())

        test[TARGET] = test_pred

        test = test.sort_index()

        test.to_csv(
                'solution-nn.csv.gz',
                columns=[TARGET],
                index=True, compression='gzip')


if __name__ == '__main__':
    main()
