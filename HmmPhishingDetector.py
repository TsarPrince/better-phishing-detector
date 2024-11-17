
import numpy as np
from hmmlearn import hmm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

class HMMPhishingDetector:
    def __init__(self, n_components=2, n_features=1000):
        self.n_components = n_components
        self.n_features = n_features
        self.vectorizer = CountVectorizer(max_features=n_features)

        # Initialize two HMM models - one for phishing and one for legitimate
        self.hmm_phishing = hmm.MultinomialHMM(n_components=n_components)
        self.hmm_legitimate = hmm.MultinomialHMM(n_components=n_components)

    def prepare_sequence_data(self, X, fit=False):
        # Convert text data to sequences of word indices
        if fit:
            X_vec = self.vectorizer.fit_transform(X).toarray()
        else:
            X_vec = self.vectorizer.transform(X).toarray()
        # Reshape for HMM (n_samples, n_timesteps, n_features)
        return X_vec.reshape(-1, 1, self.n_features)

    def fit(self, X, y):
        # Prepare data - fit and transform during training
        X_sequences = self.prepare_sequence_data(X, fit=True)

        # Split data into phishing and legitimate
        X_phish = X_sequences[y == 0]
        X_legit = X_sequences[y == 1]

        # Initialize and set starting probabilities
        startprob_phish = np.array([0.6, 0.4])  # Example starting probabilities
        startprob_legit = np.array([0.4, 0.6])

        self.hmm_phishing.startprob_ = startprob_phish
        self.hmm_legitimate.startprob_ = startprob_legit

        # Fit HMM models with proper input shape
        self.hmm_phishing.fit(X_phish.reshape(-1, self.n_features))
        self.hmm_legitimate.fit(X_legit.reshape(-1, self.n_features))

        return self

    def predict(self, X):
        # Transform only (don't fit) for prediction
        X_sequences = self.prepare_sequence_data(X, fit=False)
        predictions = []

        for sequence in X_sequences:
            # Calculate log probability for both models
            score_phishing = self.hmm_phishing.score(sequence)
            score_legitimate = self.hmm_legitimate.score(sequence)

            # Classify based on higher probability
            predictions.append(1 if score_legitimate > score_phishing else 0)

        return np.array(predictions)

    def evaluate(self, X_test, y_test):
        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        return accuracy, report, conf_matrix
