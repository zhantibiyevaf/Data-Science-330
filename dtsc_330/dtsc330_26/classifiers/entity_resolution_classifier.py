"""This is our resuable classifier. A classifier should be able
to be trained, as well as to able to make predictions. These
functions share the same data, which suggests we should use:
"""

import numpy as np
import sklearn

# import xgboost

# NOTE: We can use a custom classifier, but most classiifers are
# implemented well.
# Common classifiers (though not the best) are in scikit-learn
# Some of the best ones are their own, such as xgboost


class EntityResolutionClassifier:
    def __init__(self, model_type: str = "logistic_regression"):
        # self has to be passed within classes, it gives us access to shared variables
        # model_type is the argument
        # : str is type hinting, which shows that the input MUST be a string
        # = 'logistic_regression' is the default argument
        # ALL shared variables must be defined init
        """Create a reusable classifier

        Args:
            model_type (str, optional): model type can be either
                logistic_regression or random_forest. Defaults to
                'logistic_regression'.
        """
        self.model = None

    def train(self, features, labels):
        """Train a model. In this case, scaling is not necessary"""
        self.model = sklearn.ensemble.RandomForestClassifier()
        self.model.fit(features, labels.astype(int))

    def predict(self, features):
        """Predict labels using model_type from features

        Args:
            features (_type_): Features MUST be the same types as in
                the training data, and in the same order
        """
        features = self.scaler.transform(features)
        return self.model.predict_proba(features)[:, 1]  # we can use predict_proba
        # which returns the probability rather than true/false
        # which returns the probability rather than true/false
