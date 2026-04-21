# ==========================================
# 🔥 FIX FOR MAC (VERY IMPORTANT)
# ==========================================
import os
import zipfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


class SavedModelSentimentWrapper:
    def __init__(self, export_dir):
        self.model = tf.saved_model.load(export_dir)
        self.signature = self.model.signatures["serving_default"]
        _, keyword_args = self.signature.structured_input_signature
        self.input_key = next(iter(keyword_args))
        self.output_key = next(iter(self.signature.structured_outputs))

    def predict(self, texts, verbose=0):
        del verbose
        inputs = texts if tf.is_tensor(texts) else tf.constant(texts)
        result = self.signature(**{self.input_key: inputs})
        return result[self.output_key].numpy()


def load_sentiment_dl_model():
    keras_model_path = "sentiment/model.keras"
    saved_model_path = "sentiment/saved_dl_model"
    load_errors = []

    if os.path.isfile(keras_model_path) and zipfile.is_zipfile(keras_model_path):
        try:
            return tf.keras.models.load_model(keras_model_path)
        except Exception as exc:
            load_errors.append(f"{keras_model_path}: {exc}")

    if os.path.isdir(saved_model_path):
        try:
            return SavedModelSentimentWrapper(saved_model_path)
        except Exception as exc:
            load_errors.append(f"{saved_model_path}: {exc}")

    error_message = "No compatible sentiment deep learning model could be loaded."
    if load_errors:
        error_message += "\n" + "\n".join(load_errors)
    raise RuntimeError(error_message)

# ==========================================
# IMPORTS
# ==========================================
from translation.classical import ClassicalTranslator
from translation.seq2seq import DLTranslator
from translation.transformer_mt import TransformerTranslator

from sentiment.classical import ClassicalSentimentModel
from sentiment.bert import BertSentimentModel

from summarization.textrank import ExtractiveSummarizer
from summarization.t5_summarizer import AbstractiveSummarizer


# ==========================================
# SENTIMENT
# ==========================================
def run_sentiment():
    print("\n⏳ Loading Sentiment Models...")

    classical = ClassicalSentimentModel()
    classical.train(
        ["I love this", "I hate this", "Amazing", "Terrible"],
        [1, 0, 1, 0]
    )

    bert = BertSentimentModel()

    dl_model = load_sentiment_dl_model()

    print("✅ Models Loaded!\n")

    while True:
        text = input("Enter text (or 'back'): ")
        if text.lower() == 'back':
            break

        dl_pred = float(dl_model.predict(tf.constant([text]), verbose=0)[0][0])
        dl_sent = "Positive" if dl_pred >= 0.5 else "Negative"

        print("\n--- Sentiment Results ---")
        print(f"Classical (TF-IDF): {classical.predict(text)}")
        print(f"Deep Learning (LSTM): {dl_sent} ({dl_pred:.2f})")
        print(f"Transformer (BERT): {bert.predict(text)}")
        print("---------------------------\n")


# ==========================================
# TRANSLATION
# ==========================================
def run_translation():
    print("\n⏳ Loading Translation Models...")

    classical = ClassicalTranslator()
    dl_model = DLTranslator()
    transformer = TransformerTranslator()

    print("✅ Models Loaded!\n")

    try:
        while True:
            text = input("Enter English text (or 'back'): ")

            if text.lower() == "back":
                break

            print("\n--- Translation Results ---")
            print(f"Classical (Rule-Based): {classical.translate(text)}")
            print(f"Deep Learning (LSTM): {dl_model.translate(text)}")
            print(f"Transformer (MarianMT): {transformer.translate(text)}")
            print("----------------------------\n")
    finally:
        transformer.close()


# ==========================================
# SUMMARIZATION
# ==========================================
def run_summarization():
    print("\n⏳ Loading Summarization Models...")

    extractive = ExtractiveSummarizer()
    abstractive = AbstractiveSummarizer()

    print("✅ Models Loaded!\n")

    while True:
        text = input("Enter text (or 'back'): ")
        if text.lower() == "back":
            break

        print("\n--- Summarization Results ---")
        print(f"Classical (TextRank): {extractive.summarize(text)}")
        print(f"Transformer (T5): {abstractive.summarize(text)}")
        print("------------------------------\n")


# ==========================================
# MAIN
# ==========================================
def main():
    while True:
        print("\n" + "=" * 50)
        print("🚀 COMPARATIVE MULTITASK NLP SYSTEM")
        print("=" * 50)
        print("1. Sentiment Analysis")
        print("2. Machine Translation")
        print("3. Text Summarization")
        print("4. Exit")

        choice = input("\nSelect option (1-4): ")

        if choice == '1':
            run_sentiment()
        elif choice == '2':
            run_translation()
        elif choice == '3':
            run_summarization()
        elif choice == '4':
            print("\nExiting... 🚀")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
