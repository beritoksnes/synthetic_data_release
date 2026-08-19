""" Some predictive models to represent a simple analysis task. """
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, 
                             balanced_accuracy_score, 
                             f1_score,
                             roc_auc_score, 
                             average_precision_score,
                             mean_squared_error, 
                             mean_absolute_error)

from utils.logging import LOGGER
from utils.constants import *


class PredictiveModel(object):
    """ A predictive model. """
    def __init__(self, metadata, labelCol):
        """
        :param metadata: dict: Metadata description
        :param labelCol: str: Name of the target variable
        """
        self.metadata = metadata
        self.labelCol = labelCol
        self.nfeatures = self._get_num_features()

        self.ImputerCat = SimpleImputer(strategy='most_frequent')
        self.ImputerNum = SimpleImputer(strategy='median')

        self.datatype = pd.DataFrame
        self.trained = False

    def train(self, data):
        return NotImplementedError("Method needs to be overwritten by a subclass")

    def predict(self, features):
        return NotImplementedError("Method needs to be overwritten by a subclass")

    def evalute(self, data):
        return NotImplementedError("Method needs to be overwritten by a subclass")

    def _encode_data(self, data):
        n_samples = len(data)
        features_encoded = np.empty((n_samples, self.nfeatures))
        cidx = 0

        for cdict in self.metadata['columns']:
            data_type = cdict['type']
            attr_name = cdict['name']
            if attr_name != self.labelCol:
                col_data = data[attr_name].to_numpy()

                if data_type == FLOAT or data_type == INTEGER:
                    col_max = cdict['max']
                    col_min = cdict['min']
                    features_encoded[:, cidx] = np.true_divide(col_data - col_min, col_max + ZERO_TOL)
                    cidx += 1

                elif data_type == CATEGORICAL or data_type == ORDINAL:
                    # One-hot encoded categorical columns
                    col_cats = cdict['i2s']
                    col_data_onehot = self._one_hot(col_data, col_cats)
                    features_encoded[:, cidx : cidx + len(col_cats)] = col_data_onehot
                    cidx += len(col_cats)

        return features_encoded

    def _get_num_features(self):
        nfeatures = 0

        for cdict in self.metadata['columns']:
            data_type = cdict['type']
            attr_name = cdict['name']

            if attr_name != self.labelCol:
                if data_type == FLOAT or data_type == INTEGER:
                    nfeatures += 1

                elif data_type == CATEGORICAL or data_type == ORDINAL:
                    nfeatures += len(cdict['i2s'])

                else:
                    raise ValueError(f'Unkown data type {data_type} for attribute {attr_name}')

        return nfeatures

    def _get_feature_names(self):
        featureNames = []

        for i, cdict in enumerate(self.metadata['columns']):
            data_type = cdict['type']
            attr_name = cdict['name']

            if attr_name != self.labelCol:
                if data_type == FLOAT or data_type == INTEGER:
                    featureNames.append(attr_name)

                elif data_type == CATEGORICAL or data_type == ORDINAL:
                    col_cats = cdict['i2s']
                    featureNames.extend([f'{attr_name}_{c}' for c in col_cats])

        return featureNames

    def _impute_missing_values(self, df):
        dfImpute = df.copy()

        catCols = []
        numCols = []

        for col in self.metadata['columns']:
            if col['name'] in list(dfImpute):
                if col['type'] in [CATEGORICAL, ORDINAL]:
                    catCols.append(col['name'])
                elif col['type'] in NUMERICAL:
                    numCols.append(col['name'])

        self.ImputerCat.fit(df[catCols])
        dfImpute[catCols] = self.ImputerCat.transform(df[catCols])

        self.ImputerNum.fit(df[numCols])
        dfImpute[numCols] = self.ImputerNum.transform(df[numCols])

        return dfImpute

    def _one_hot(self, col_data, categories):
        col_data_onehot = np.zeros((len(col_data), len(categories)))
        cidx = [categories.index(c) for c in col_data]
        col_data_onehot[np.arange(len(col_data)), cidx] = 1

        return col_data_onehot


class ClassificationTask(PredictiveModel):
    """ A binary or multiclass classification model. """

    def __init__(self, Distinguisher, metadata, labelCol):
        """
        :param Distinguisher: sklearn.Classifier: A classification model
        :param metadata: dict: Metadata description
        :param labelCol: str: Name of the target variable
        """
        super().__init__(metadata, labelCol)
        self.Distinguisher = Distinguisher

        labels = self._get_labels()
        self.labels = {l:i for i, l in enumerate(labels)}
        self.labelsInv = {i:l for l, i in self.labels.items()}

        self.__name__ = f'{self.Distinguisher.__class__.__name__}{self.labelCol}'

    def train(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        data = self._impute_missing_values(data)
        features = self._encode_data(data.drop(self.labelCol, axis=1))
        labels = data[self.labelCol].apply(lambda x: self.labels[x]).values

        self.Distinguisher.fit(features, labels)

        LOGGER.debug('Finished training MIA distinguisher')
        self.trained = True

    def predict(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        features = self._encode_data(data.drop(self.labelCol, axis=1))
        labels = self.Distinguisher.predict(features)

        return [self.labelsInv[i] for i in labels]

    def evaluate(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        features = self._encode_data(data.drop(self.labelCol, axis=1))
        labelsTrue = data[self.labelCol].apply(lambda x: self.labels[x]).values
        labelsPred = self.Distinguisher.predict(features)

        return [int(l == p) for l, p in zip(labelsTrue, labelsPred)]

    def _predict_proba_full(self, features):
        """ Probability matrix over the *full* label space from metadata.

        Distinguisher.classes_ only holds the classes actually seen during
        training, which can be a strict subset when training on synthetic
        data that failed to reproduce a rare class. Columns for unseen
        classes are filled with zeros, so rows still sum to 1.
        """
        allCodes = sorted(self.labelsInv.keys())
        probaSeen = self.Distinguisher.predict_proba(features)

        proba = np.zeros((features.shape[0], len(allCodes)))
        for j, code in enumerate(self.Distinguisher.classes_):
            proba[:, allCodes.index(code)] = probaSeen[:, j]

        return proba, allCodes

    def get_metrics(self, data, posLabel=None):
        """ Aggregate performance metrics on a held-out set.

        :param data: DataFrame: Evaluation data (real test set)
        :param posLabel: The positive class for binary metrics. Defaults to
                         the second category in metadata['i2s'].
        :return: dict: metric name -> scalar
        """
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        if not self.trained:
            raise RuntimeError('Model must be trained before evaluation')

        features = self._encode_data(data.drop(self.labelCol, axis=1))
        labelsTrue = data[self.labelCol].apply(lambda x: self.labels[x]).values
        labelsPred = self.Distinguisher.predict(features)
        proba, allCodes = self._predict_proba_full(features)

        results = {
            'Accuracy': accuracy_score(labelsTrue, labelsPred),
            'BalancedAccuracy': balanced_accuracy_score(labelsTrue, labelsPred),
            'nClassesTrain': len(self.Distinguisher.classes_),
        }

        if len(allCodes) == 2:
            posCode = self.labels[posLabel] if posLabel is not None else allCodes[1]
            binTrue = (labelsTrue == posCode).astype(int)
            scores = proba[:, allCodes.index(posCode)]

            results['F1'] = f1_score(labelsTrue, labelsPred, pos_label=posCode,
                                     zero_division=0)
            if 0 < binTrue.sum() < len(binTrue):
                results['AUC-ROC'] = roc_auc_score(binTrue, scores)
                results['AUC-PR'] = average_precision_score(binTrue, scores)
            else:
                results['AUC-ROC'] = np.nan   # only one class present in test set
                results['AUC-PR'] = np.nan
        else:
            results['F1'] = f1_score(labelsTrue, 
                                     labelsPred, 
                                     average='macro',
                                     zero_division=0)
            
            aucs = []
            for code in allCodes:
                binTrue = (labelsTrue == code).astype(int)
                if 0 < binTrue.sum() < len(binTrue):
                    aucs.append(roc_auc_score(binTrue, proba[:, allCodes.index(code)]))
            results['AUC-ROC'] = np.mean(aucs) if aucs else np.nan

        return results

    def _get_accuracy(self, trueLabels, predLabels):
        return sum([g == l for g, l in zip(trueLabels, predLabels)])/len(trueLabels)

    def _get_labels(self):
        for cdict in self.metadata['columns']:
            if cdict['name'] == self.labelCol:
                if not cdict['type'] in [CATEGORICAL, ORDINAL]:
                    raise ValueError('Label column must be discrete data type.')

                return cdict['i2s']


class RandForestClassTask(ClassificationTask):
    def __init__(self, metadata, labelCol):
        super().__init__(RandomForestClassifier(), metadata, labelCol)


class LogRegClassTask(ClassificationTask):
    def __init__(self, metadata, labelCol):
        super().__init__(LogisticRegression(), metadata, labelCol)


class RegressionTask(PredictiveModel):
    """ A regression model. """

    def __init__(self, Regressor, metadata, labelCol):
        """

        :param Regressor: sklearn.Regressor: A regression model
        :param metadata: dict: Metadata description
        :param labels: list: Label names
        :param FeatureSet: object: Feature extraction object
        """
        super().__init__(metadata, labelCol)
        self.Regressor = Regressor

        self.__name__ = f'{self.Regressor.__class__.__name__}{self.labelCol}'

    def train(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        data = self._impute_missing_values(data)
        features = self._encode_data(data.drop(self.labelCol, axis=1))
        labels = data[self.labelCol].values

        self.Regressor.fit(features, labels)

        LOGGER.debug('Finished training regression model')
        self.trained = True

    def predict(self, features):
        if not isinstance(features, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(features)}")

        features = self._encode_data(features)
        labels = self.Regressor.predict(features)

        return list(labels)

    def evaluate(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        features = self._encode_data(data.drop(self.labelCol, axis=1))
        labelsTrue = data[self.labelCol].values
        labelsPred = self.Regressor.predict(features)

        return [true - pred for true, pred in zip(labelsTrue, labelsPred)]

    def get_metrics(self, data):
        if not isinstance(data, self.datatype):
            raise ValueError(f"Model expects input as {self.datatype} but got {type(data)}")

        if not self.trained:
            raise RuntimeError('Model must be trained before evaluation')

        features = self._encode_data(data.drop(self.labelCol, axis=1))
        labelsTrue = data[self.labelCol].values
        labelsPred = self.Regressor.predict(features)

        return {'MSE': mean_squared_error(labelsTrue, labelsPred),
                'MAE': mean_absolute_error(labelsTrue, labelsPred)}


class LinRegTask(RegressionTask):
    def __init__(self, metadata, labelCol):
        super().__init__(LinearRegression(), metadata, labelCol)
