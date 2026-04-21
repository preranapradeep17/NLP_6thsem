from transformers import pipeline

class BertSentimentModel:
    def __init__(self):
        print("⏳ Loading BERT model (this may take a few seconds)...")
        self.nlp_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        print("✅ BERT Model Loaded!")

    def predict(self, text):
        """Predict sentiment using BERT"""
        result = self.nlp_pipeline(text)[0]

        label = "Positive" if result['label'] == 'POSITIVE' else "Negative"
        return f"{label} (Confidence: {round(result['score'], 2)})"