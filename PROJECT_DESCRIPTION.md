## NLP Tasks Web App - Project Description

**A Flask-powered web application for core NLP tasks:**

- **Sentiment Analysis**: Classical ML, BERT (PyTorch), LSTM/TF-IDF (TensorFlow/Keras).
- **Machine Translation (EN-FR)**: Classical, Seq2Seq RNN (pre-trained), Transformer (HuggingFace).
- **Text Summarization**: TextRank (extractive), T5 (abstractive).

**Key Features:**
- Unified API endpoints (`/api/sentiment`, `/api/translate`, `/api/summarize`).
- Static HTML frontend (`static/index.html`).
- Pre-trained models included (sentiment LSTM, Seq2Seq, tokenizers).
- Lazy-loading for PyTorch/TF models; subprocess isolation for TensorFlow.
- Runs on `python app.py` → http://127.0.0.1:5050.

**Tech Stack:** Flask, PyTorch, TensorFlow/Keras, Transformers, NLTK, scikit-learn.

**Quick Start:**
```
pip install -r requirements.txt
python app.py
```

**Status:** Production-ready local demo with multiple classical/DL models benchmarked.

