#!/usr/bin/env python3

import random
import pandas as pd
import numpy as np
import logging
import argparse
import common

import tensorflow as tf

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

    @common.log_duration
    def fit(self, train, test=None):
        encoders = {}
        variables = list(train.columns)
        for var in variables:
            uniq = np.unique(train[var])
            if len(uniq) <= 2 or len(uniq) >= 105:
                continue
            encoder = OneHotEncoder(
                    sparse=False, handle_unknown='ignore', dtype='int8')
            encoder.fit(train[[var]]+1)
            encoders[var] = encoder
        self.encoders = encoders

    @common.log_duration
    def predict(self, X):
        X = X.copy(deep=False)
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

    def compute_gini(self, data):
        data_X, data_y = data

        y_pred_val = self.model.predict(data_X, batch_size=2048)
        return common.CostFunction(data_y, y_pred_val.reshape([-1]))

    def on_epoch_end(self, epoch, logs={}):
        val_gini = self.compute_gini(self.val)
        train_gini = self.compute_gini(self.train)
        print('\ngini-train: {:.5f} gini-val: {:.5f}'.format(
            train_gini, val_gini))

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass


def NNModelFactory(target, epochs, seed=common.SEED):
    return lambda: NNModel(target, epochs, seed)


class NNModel:
    def __init__(self, target, epochs=10, seed=common.SEED):
        self.target = target
        self.epochs = epochs
        self.seed = seed

    @common.log_duration
    def filter_data(self, df):
        to_drop = {
                'ps_car_11_cat', 'ps_ind_14', 'ps_car_11', 'ps_car_14',
                'ps_ind_06_bin', 'ps_ind_09_bin', 'ps_ind_10_bin',
                'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'}

        cols_use = [
                c for c in df.columns if (not c.startswith('ps_calc_'))
                and (c not in to_drop)]
        return df[cols_use].copy(deep=False)

    @common.log_duration
    def make_model(self, input_size, hidden_units):
        model = Sequential()
        model.add(Dense(
            units=hidden_units, activation='relu', input_dim=input_size))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))
        model.add(Activation('sigmoid'))

        optimizer = Adam(lr=0.001)

        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

    @common.log_duration
    def upsample(self, train):
        positive = train[self.target] == 1
        return pd.concat([train] + [train[positive]]*4)

    @common.log_duration
    def fit(self, train, test=None):
        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)

        train = self.upsample(train)

        train = self.filter_data(train)

        train_X = np.array(train.drop(self.target, axis=1), dtype='float32')
        train_y = np.array(train[self.target], dtype='float32')
        del train

        self.model = self.make_model(train_X.shape[1], 35)

        validation_data = None
        if test is not None:
            test = self.filter_data(test)
            validation_data = (
                    np.array(test.drop(self.target, axis=1), dtype='float32'),
                    np.array(test[self.target], dtype='float32'))
            del test

        self.model.fit(
                train_X,
                train_y,
                batch_size=2048,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=[GiniCallback(
                    self.target, (train_X, train_y), validation_data)])

    @common.log_duration
    def predict(self, X):
        X = self.filter_data(X)
        if self.target in X.columns:
            X = X.drop(self.target, axis=1)
        return self.model.predict(
                np.array(X, dtype='float32'), batch_size=2048).reshape([-1])


def MakeModelFactory(target, nn_params):
    rand = random.Random(common.SEED)
    factories = [NNModelFactory(
        target, seed=rand.randint(1, 1000000), **nn_params)
        for i in range(3)]
    base_model = common.NestedClassifiersFactory([
        common.AverageClassifierFactory(*factories),
    ])
    return common.NestedClassifiersFactory([
        lambda: SpecializedFeatures(target),
        # lambda: common.FeatureScaler(target),
        lambda: common.CrossValidator(target, base_model),
    ])


@common.log_duration
def SaveDF(df, path, columns):
    df.to_csv(
            path,
            columns=columns,
            index=True,
            compression='gzip')


def main():
    parser = argparse.ArgumentParser()
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

    TARGET = 'target'

    nn_params = dict(epochs=11)

    if args.search:
        params_space = dict(
            epochs=[10, 11, 12, 13, 14, 15],
        )

        def ModelFactory(**params):
            params = dict(params)
            p = dict(nn_params)
            p.update(params)
            return MakeModelFactory(TARGET, p)()
        model = common.HyperparamSearch(ModelFactory, params_space)
        model.fit(train)

        for score, params in sorted(model.history, key=lambda x: x[0]):
            logging.info('='*80)
            logging.info('Score: %f', score)
            logging.info('Params: %s', params)

    elif args.full:
        factory = MakeModelFactory(TARGET, nn_params)
        logging.info('Training...')
        model = factory()
        model.fit(train)
    else:
        print('No action requested')
        return

    logging.info('CV Score: %f', model.loss)
    logging.info('Evaluating test set...')
    test = gen_features(load_test_data())
    test_pred = model.predict(test)

    logging.info('Prediction mean: %f', test_pred.mean())
    logging.info('Train mean: %f', train[TARGET].mean())

    test[TARGET] = test_pred
    train[TARGET] = model.train_preds

    common.SaveDF(test, 'solution-nn.csv.gz', columns=[TARGET])
    common.SaveDF(train, 'train-nn.csv.gz', columns=[TARGET])


if __name__ == '__main__':
    main()
