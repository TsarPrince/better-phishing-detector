
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


class MEMMPhishingDetector:
    def __init__(self, n_features=1000):
        self.n_features = n_features
        self.vectorizer = TfidfVectorizer(max_features=n_features)
        self.weights = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feature_function(self, x, y):
        # Combine email features with previous state
        return np.concatenate([x, [y]])

    def calculate_likelihood(self, weights, X, y):
        total_likelihood = 0
        for i in range(1, len(X)):
            features = self.feature_function(X[i], y[i-1])
            prob = self.sigmoid(np.dot(weights, features))
            total_likelihood += y[i] * np.log(prob) + (1 - y[i]) * np.log(1 - prob)
        return -total_likelihood  # Negative for minimization

    def fit(self, X, y):
        # Transform text data
        X_transformed = self.vectorizer.fit_transform(X).toarray()

        # Initialize weights
        initial_weights = np.zeros(self.n_features + 1)  # +1 for previous state

        # Optimize weights using L-BFGS-B
        result = optimize.minimize(
            fun=self.calculate_likelihood,
            x0=initial_weights,
            args=(X_transformed, y),
            method='L-BFGS-B'
        )

        self.weights = result.x
        return self

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X).toarray()
        predictions = []
        prev_state = 0  # Initial state

        for x in X_transformed:
            features = self.feature_function(x, prev_state)
            prob = self.sigmoid(np.dot(self.weights, features))
            pred = 1 if prob > 0.5 else 0
            predictions.append(pred)
            prev_state = pred

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return accuracy, report, conf_matrix
