# ==========================================
# 🔥 SUPPRESS ALL TF / HF WARNINGS FIRST
# ==========================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

import warnings
warnings.filterwarnings("ignore")

import logging
logging.disable(logging.WARNING)

# ==========================================
# FLASK APP
# ==========================================
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')
app.config['JSON_SORT_KEYS'] = False

# ==========================================
# EAGER FIRE PYTORCH MODELS TO SEIZE MACOS THREADPOOL 
# ==========================================
print("\n⏳ Pre-loading PyTorch Models to bypass macOS thread restrictions...")
from sentiment.bert import BertSentimentModel
from translation.transformer_mt import TransformerTranslator
from summarization.t5_summarizer import AbstractiveSummarizer
from summarization.textrank import ExtractiveSummarizer
from translation.classical import ClassicalTranslator
from sentiment.classical import ClassicalSentimentModel

_bert = BertSentimentModel()
_transformer = TransformerTranslator()
_abstractive = AbstractiveSummarizer()

_extractive = ExtractiveSummarizer()
_classic_trans = ClassicalTranslator()
_classic_sent = ClassicalSentimentModel()
_classic_sent.train(
    ["I love this", "I hate this", "Amazing", "Terrible", "great job", "awful experience", 
     "fantastic", "horrible", "excellent", "poor quality", "wonderful", "disgusting",
     "best ever", "worst ever", "highly recommend", "do not buy", "very happy", "very sad", 
     "brilliant", "dreadful"],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
)

print("✅ PyTorch and Classical Models loaded successfully.")

# ==========================================
# LAZY LOAD TENSORFLOW
# ==========================================
_models: dict = {}

def get_sentiment_models():
    if 'sentiment' not in _models:
        print("\n⏳ Initializing Sentiment models...")
        class SubprocessTFSentiment:
            import subprocess
            import sys
            def predict(self, texts, verbose=0):
                try:
                    out = self.subprocess.check_output(
                        [self.sys.executable, "predict_tf.py", texts[0]], 
                        stderr=self.subprocess.DEVNULL
                    ).decode("utf-8").strip()
                    return [[float(out)]]
                except Exception:
                    return [[0.5]]
                    
        dl_sent = SubprocessTFSentiment()
        _models['sentiment'] = {
            'classical': _classic_sent,
            'bert': _bert,
            'dl': dl_sent,
        }
        print("✅ Sentiment models ready.")
    return _models['sentiment']

def get_translation_models():
    if 'translation' not in _models:
        print("\n⏳ Initializing Translation models...")
        class SubprocessDLTranslator:
            import subprocess
            import sys

            def translate(self, text):
                try:
                    child_env = os.environ.copy()
                    child_env.pop('TF_USE_LEGACY_KERAS', None)
                    out = self.subprocess.check_output(
                        [self.sys.executable, "predict_translation.py", text],
                        stderr=self.subprocess.DEVNULL,
                        env=child_env,
                    ).decode("utf-8").strip()
                    lines = [line.strip() for line in out.splitlines() if line.strip()]
                    return lines[-1] if lines else "[DL translation produced no output.]"
                except Exception as exc:
                    return f"[DL translation unavailable: {exc}]"

        _models['translation'] = {
            'classical': _classic_trans,
            'transformer': _transformer,
            'dl': SubprocessDLTranslator(),
        }
        print("✅ Translation models ready.")
    return _models['translation']

def get_summarization_models():
    if 'summarization' not in _models:
        _models['summarization'] = {
            'extractive': _extractive,
            'abstractive': _abstractive,
        }
    return _models['summarization']


# ==========================================
# HELPER
# ==========================================
def _get_text(req) -> tuple:
    """Extract and validate text from JSON body. Returns (text, error_response)."""
    data = req.get_json(silent=True) or {}
    text = str(data.get('text', '')).strip()
    if not text:
        return None, (jsonify({'error': 'No text provided'}), 400)
    return text, None


# ==========================================
# ROUTES — Static
# ==========================================
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ==========================================
# ROUTES — API
# ==========================================
@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    text, err = _get_text(request)
    if err:
        return err

    try:
        m = get_sentiment_models()

        # --- Deep Learning (LSTM) ---
        dl_raw = float(m['dl'].predict([text], verbose=0)[0][0])
        dl_label = "Positive" if dl_raw >= 0.5 else "Negative"
        dl_conf  = round(dl_raw if dl_raw >= 0.5 else 1 - dl_raw, 4)

        # --- BERT ---
        bert_res   = m['bert'].nlp_pipeline(text)[0]
        bert_label = "Positive" if bert_res['label'] == 'POSITIVE' else "Negative"
        bert_conf  = round(bert_res['score'], 4)

        # --- Classical (TF-IDF + LR) ---
        classical_label = m['classical'].predict(text)

        return jsonify({
            'classical': {'label': classical_label, 'confidence': None},
            'dl':        {'label': dl_label,        'confidence': dl_conf},
            'bert':      {'label': bert_label,      'confidence': bert_conf},
        })

    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/translate', methods=['POST'])
def translate():
    text, err = _get_text(request)
    if err:
        return err

    try:
        m = get_translation_models()
        return jsonify({
            'classical':   m['classical'].translate(text),
            'dl':          m['dl'].translate(text),
            'transformer': m['transformer'].translate(text),
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/summarize', methods=['POST'])
def summarize():
    text, err = _get_text(request)
    if err:
        return err

    # Summarization needs at least 2 sentences
    if len(text.split('.')) < 2 or len(text.split()) < 20:
        return jsonify({
            'error': 'Please enter at least 2–3 sentences (20+ words) for meaningful summarization.'
        }), 400

    try:
        m = get_summarization_models()
        extractive  = m['extractive'].summarize(text)
        abstractive = m['abstractive'].summarize(text)
        return jsonify({
            'extractive':  extractive  or 'Could not extract summary.',
            'abstractive': abstractive or 'Could not generate summary.',
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  🚀  NLP Web App  →  http://127.0.0.1:5050")
    print("=" * 55 + "\n")
    app.run(debug=False, host='127.0.0.1', port=5050, use_reloader=False, threaded=False)
