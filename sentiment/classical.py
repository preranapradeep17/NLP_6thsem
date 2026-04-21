from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class ClassicalSentimentModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.is_trained = False

    def train(self, texts, labels):
        """Train TF-IDF + Logistic Regression model"""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        print("✅ Classical Model Trained!")

    def predict(self, text):
        """Predict sentiment"""
        if not self.is_trained:
            return "Error: Model not trained yet."

        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]

        return "Positive" if prediction == 1 else "Negative"