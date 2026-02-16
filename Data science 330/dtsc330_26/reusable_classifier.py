"""This is our resuable classifier. A classifier should be able
to be trained, as well as to able to make predictions. These
functions share the same data, which suggests we should use:
"""
import numpy as np
import sklearn

# NOTE: We can use a custom classifier, but most classiifers are 
# implemented well.
# Common classifiers (though not the best) are in scikit-learn
# Some of the best ones are their own, such as xgboost


class ReusableClassifier:
    def __init__(self, model_type: str = 'logistic_regression'):
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
        self.model_type = model_type
        self.scaler = None

    def train(self, features, labels):
        if self.model_type == 'logistic_regression':
            self.model = sklearn.linear_model.LogisticRegression(max_iter=1000)

        elif self.model_type == 'random_forest':
            self.model = sklearn.ensemble.RandomForestClassifier()
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(features)
        features_scaled = self.scaler.transform(features)

        # fit the model
        self.model.fit(features_scaled, labels.astype(int))


    def predict(self, features):
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
        
        

    def predict(self, features):
        """Predict labels using model_type from features

        Args:
            features (_type_): Features MUST be the same types as in
                the training data, and in the same order
        """
        features = self.scaler.transform(features)
        return self.model.predict(features)  # we can use predict_proba
        # which returns the probability rather than true/false

    def assess(self, features, labels, random_number: int = 42):
        """Assess how well the classifier performed."""
        # Train test split
        # If you ask how well the model did on the training data
        # that's different from how well the model performs
        # No statistician will accept it
        train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(features, labels, test_size=0.2, random_state=random_number)
        self.train(train_features, train_labels)

        predictions = self.predict(test_features)
        return 1 + np.sum(predictions.astype(float) - test_labels.to_numpy().astype(float))/len(test_labels)

    def save(self, path: str):
        """Save our model to the location path."""
        pass

    def load(self, path: str):
        """Load in a model from a path."""
        pass



if __name__ == '__main__':
    rc = ReusableClassifier()
