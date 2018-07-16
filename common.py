import gc
import lightgbm as lgb
import logging
import numpy as np
import pandas as pd
import random
import time

from numba import jit
from sklearn.model_selection import StratifiedKFold

SEED = 8797987

FOLDS = 5


def log_duration(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info(
                'Function %s started', func.__qualname__)
        ret = func(*args, **kwargs)
        logging.info(
                'Function %s took %fs', func.__qualname__, time.time() - start)
        return ret
    return wrapper


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


def merge_dicts(d1, d2):
    for k, v in d2.items():
        if k not in d1:
            d1[k] = v
        else:
            d1[k] += v


class SKLearnWrapperFactory:
    def __init__(self, target, classifier_factory):
        self.target = target
        self.factory = classifier_factory

    def __call__(self):
        return SKLearnWrapper(self.target, self.factory())


class SKLearnWrapper:
    def __init__(self, target, classifier):
        self.target = target
        self.classifier = classifier

    @log_duration
    def fit(self, train, val=None):
        self.classifier.fit(
                train.drop(self.target, axis=1), train[self.target])

    @log_duration
    def predict(self, X):
        if self.target in X.columns:
            X = X.drop(self.target, axis=1)
        return self.classifier.predict_proba(X)[:, 1]


class NestedClassifiersFactory:
    def __init__(self, factories):
        self.factories = factories

    def __call__(self):
        return NestedClassifiers([factory() for factory in self.factories])


class NestedClassifiers:
    def __init__(self, classifiers):
        self._classifiers = classifiers

    def fit(self, train, validation=None):
        for i, classifier in enumerate(self._classifiers):
            classifier.fit(train, validation)
            if i < len(self._classifiers) - 1:
                train = classifier.predict(train)
                if validation is not None:
                    validation = classifier.predict(validation)

    @property
    def loss(self):
        return self._classifiers[-1].loss

    @property
    def train_preds(self):
        return self._classifiers[-1].train_preds

    def predict(self, X):
        for classifier in self._classifiers:
            X = classifier.predict(X)
        return X


class LGBMFactory:
    def __init__(self, target, **args):
        self.target = target
        self.args = args

    def __call__(self):
        return LGBM(self.target, **self.args)


class LGBM:
    def __init__(
            self, target, num_rounds=100, eta=0.3, min_child_weight=None,
            subsample=None, colsample=None, num_leaves=None,
            lambda_l1=None, lambda_l2=None, scale_pos_weight=None,
            min_split_gain=None, max_depth=None, early_stopping=None):
        self.target = target
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
        self.max_depth = max_depth
        self.early_stopping = early_stopping

    def _MakeArgs(self, train_set, evals=None):
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
            'max_depth': self.max_depth,
        }

        params = {k: v for k, v in params.items() if v is not None}

        args = dict(
            params=params,
            train_set=train_set,
            num_boost_round=self.num_rounds,
            feval=gini_xgb,
            verbose_eval=10,
        )

        if evals:
            args['valid_sets'] = evals
        if self.early_stopping is not None:
            args['early_stopping_rounds'] = self.early_stopping

        return args

    def predict(self, X):
        if self.target in X.columns:
            X = X.drop(self.target, axis=1)
        assert self.features == list(X.columns)
        return self.gbm.predict(X, num_iteration=self.gbm.best_iteration)

    @log_duration
    def fit(self, train, test):
        self.features = list(train.columns)
        self.features.remove(self.target)

        lgb_train = lgb.Dataset(
                train.drop(self.target, axis=1), train[self.target])
        evals = None
        if self.target in test.columns:
            evals = lgb.Dataset(
                    test.drop(self.target, axis=1), test[self.target])

        args = self._MakeArgs(
                lgb_train, evals=evals)
        self.gbm = lgb.train(**args)


class Ensemble(object):
    def __init__(self, target, stacker_factory, model_factories, folds=FOLDS):
        self.target = target
        self.folds = folds
        self.stacker_factory = stacker_factory
        self.model_factories = model_factories

    @log_duration
    def fit(self, train, test=None):

        preds = pd.DataFrame()
        preds = train[[self.target]].copy()
        loss = []
        self.classifiers = []
        for i, classifier in enumerate(self.model_factories):
            cv = CrossValidator(self.target, classifier, folds=self.folds)
            cv.fit(train)
            self.classifiers.append(cv)

            preds.loc[:, 'model_{}'.format(i)] = cv.train_preds
            loss.append(cv.loss)

        print("Stacker score: %.5f" % (np.mean(loss)))

        self.stacker = CrossValidator(
                self.target, self.stacker_factory, folds=self.folds)
        self.stacker.fit(preds)

    @log_duration
    def predict(self, X):
        preds = pd.DataFrame(index=X.index)
        for i, classifier in enumerate(self.classifiers):
            pred = classifier.predict(X)
            preds.loc[:, 'model_{}'.format(i)] = pred
        print(preds.describe())
        return self.stacker.predict(preds)

    @property
    def loss(self):
        return self.stacker.loss

    @property
    def train_preds(self):
        return self.stacker.train_preds


class CrossValidator:
    def __init__(self, target, model_factory, folds=FOLDS):
        self.target = target
        self.model_factory = model_factory
        self.folds = folds

    @log_duration
    def fit(self, train, test=None):
        loss = []
        models = []

        select = StratifiedKFold(self.folds, shuffle=True, random_state=SEED)
        train_preds = np.zeros(len(train))
        for train_index, test_index in select.split(train, train[self.target]):
            sub_train = train.iloc[train_index].copy(deep=False)
            sub_val = train.iloc[test_index].copy(deep=False)

            model = self.model_factory()
            model.fit(sub_train, sub_val)

            val_pred = model.predict(sub_val)
            cv = CostFunction(sub_val[self.target], val_pred)
            loss.append(cv)
            train_preds[test_index] = val_pred

            models.append(model)

            logging.info('Fold CV: %f, Running CV mean: %f', cv, np.mean(loss))
            gc.collect()
        self.models = models
        self.loss = np.mean(loss)
        self.train_preds = train_preds

    @log_duration
    def predict(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        return sum(preds) / len(preds)


class HCCEncoder:
    def __init__(self, variable, target, min_samples_leaf, smoothing):
        self.variable = variable
        self.target = target
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = smoothing

    def fit(self, train_X, train_y=None, test_X=None, test_y=None):
        prior_prob = train_X[self.target].mean()

        hcc_name = "_".join(['hcc', self.variable, self.target])

        grouped = train_X.groupby(self.variable)
        grouped = grouped[self.target].agg(["size", "mean"])

        smoothing = 1/(1+np.exp(
            -(grouped["size"]-self.min_samples_leaf)/self.smoothing))

        grouped[hcc_name] = (
                prior_prob * (1 - smoothing) +
                grouped["mean"] * smoothing)

        self.grouped = grouped[[hcc_name]]

    def predict(self, X):
        return X.join(self.grouped, on=self.variable)


class AverageClassifierFactory:
    def __init__(self, *factories):
        self.factories = factories

    def __call__(self):
        return AverageClassifier(self.factories)


class AverageClassifier:
    def __init__(self, factories):
        self.factories = factories

    def fit(self, train, val):
        models = []
        for factory in self.factories:
            model = factory()
            model.fit(train, val)
            models.append(model)

        self.models = models

    def predict(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        return sum(preds) / len(preds)


@log_duration
def SaveDF(df, path, columns):
    df.to_csv(
            path,
            columns=columns,
            index=True,
            compression='gzip')


class UpsamplerFactory:
    def __init__(self, target, base_model_factory):
        self.factory = base_model_factory
        self.target = target

    def __call__(self):
        return Upsampler(self.target, self.factory())


class Upsampler:
    def __init__(self, target, base_model):
        self.target = target
        self.base_model = base_model

    def fit(self, train, validation=None):
        positive = train[self.target] == 1
        train = pd.concat([train] + [train[positive]]*4)
        return self.base_model.fit(train, validation)

    def predict(self, X):
        return self.base_model.predict(X)

    @property
    def loss(self):
        return self.base_model.loss

    @property
    def train_preds(self):
        return self.base_model.train_preds


class FeatureScaler:
    def __init__(self, target):
        self.target = target

    @log_duration
    def fit(self, train, validation=None):
        minmax = {}
        for feature in train.columns:
            if feature == self.target:
                continue
            col = train[feature]
            minmax[feature] = (col.min(), col.max())
        self.minmax = minmax

    @log_duration
    def predict(self, X):
        X = X.copy(deep=False)
        for (feature, (minimum, maximum)) in self.minmax.items():
            ratio = float(maximum - minimum)
            if ratio < 0.00001 or ratio == 1:
                continue
            X[feature] = (
                    X[feature].astype('float32') - minimum) * (2.0/ratio) - 1
        return X


def AveragerFactory(target, weights):
    return lambda: Averager(target, weights)


class Averager:
    def __init__(self, target, weights):
        self.target = target
        self.weights = weights

    def fit(self, train, validation):
        self.features = [f for f in train.columns if f != self.target]
        assert len(self.features) == len(self.weights)

    @log_duration
    def predict(self, X):
        W = sum(self.weights)
        return sum([X[f]*w for f, w in zip(self.features, self.weights)]) / W


def AveragerRankFactory(target, weights):
    return lambda: AveragerRank(target, weights)


class AveragerRank:
    def __init__(self, target, weights):
        self.target = target
        self.weights = weights

    def fit(self, train, validation):
        self.features = [f for f in train.columns if f != self.target]
        assert len(self.features) == len(self.weights)

    @log_duration
    def predict(self, X):
        W = sum(self.weights)
        values = []
        for f, w in zip(self.features, self.weights):
            ranked = np.argsort(np.argsort(X[f]))
            ranked = ranked * (w / len(X))
            values.append(ranked)
        return sum(values) / W


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


class HyperparamSearch:
    def __init__(self, factory, params_space):
        self.factory = factory
        self.params_space = params_space

    def fit(self, train, val=None):

        best_score = -1e100
        best_params = None
        best_model = None

        history = []

        param_combinations = product_params(self.params_space)
        logging.info(
                'Search %d param combinations...', len(param_combinations))
        for params in param_combinations:
            params = dict(params)
            logging.info('-----------------------------------------------')
            logging.info('Params: %s', params)

            model = self.factory(**params)

            model.fit(train, val)
            score = model.loss

            logging.info('Score: %f', score)
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
                logging.info('Best score so far')
            logging.info('Current best score: %f', best_score)
            logging.info('Current best params: %s', best_params)
            history.append((score, params))

        self.history = history

        self.best_model = best_model

    def predict(self, X):
        return self.best_model.predict(X)

    @property
    def loss(self):
        return self.best_model.loss

    @property
    def train_preds(self):
        return self.best_model.train_preds
