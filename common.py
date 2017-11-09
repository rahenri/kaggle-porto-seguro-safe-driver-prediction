import numpy as np
import time
import logging
import lightgbm as lgb
import pandas as pd

from numba import jit
from sklearn.model_selection import StratifiedKFold

SEED = 265359275

FOLDS = 5


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

    def fit(self, train, val=None):
        self.classifier.fit(
                train.drop(self.target, axis=1), train[self.target])

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
            min_split_gain=None, max_depth=None):
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
            'max_depth': self.max_depth,
        }

        params = {k: v for k, v in params.items() if v is not None}

        args = dict(
            params=params,
            train_set=train_set,
            num_boost_round=self.num_rounds,
            feval=gini_xgb,
            verbose_eval=0,
        )

        if evals:
            args['valid_sets'] = evals
        if early_stopping_rounds is not None:
            args['early_stopping_rounds'] = early_stopping_rounds

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
                lgb_train, evals=evals,
                early_stopping_rounds=50)
        self.gbm = lgb.train(**args)


class Ensemble(object):
    def __init__(self, target, stacker_factory, model_factories, folds=FOLDS):
        self.target = target
        self.folds = folds
        self.stacker_factory = stacker_factory
        self.model_factories = model_factories

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
        self.loss = self.stacker.loss

    def predict(self, X):
        preds = pd.DataFrame(index=X.index)
        for i, classifier in enumerate(self.classifiers):
            pred = classifier.predict(X)
            preds.loc[:, 'model_{}'.format(i)] = pred
        print(preds.describe())
        return self.stacker.predict(preds)


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
            sub_train = train.iloc[train_index].copy()
            sub_val = train.iloc[test_index].copy()

            model = self.model_factory()
            model.fit(sub_train, sub_val)

            val_pred = model.predict(sub_val)
            cv = CostFunction(sub_val[self.target], val_pred)
            loss.append(cv)
            train_preds[test_index] = val_pred

            models.append(model)

            logging.info('Fold CV: %f, Running CV mean: %f', cv, np.mean(loss))
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
    def __init__(self, base_model_factory, runs):
        self.factory = base_model_factory
        self.runs = runs

    def __call__(self):
        return AverageClassifier(self.factory, self.runs)


class AverageClassifier:
    def __init__(self, base_model_factory, runs):
        self.factory = base_model_factory
        self.runs = runs

    def fit(self, train, val):
        models = []
        for i in range(self.runs):
            model = self.factory()
            model.fit(train, val)
            models.append(model)

        self.models = models

    def predict(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict(X))
        return sum(preds) / len(preds)
