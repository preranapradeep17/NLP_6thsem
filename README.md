# NLP Tasks Web Application

## Overview
This project is a comprehensive Natural Language Processing (NLP) web application built with Flask. It provides implementations for multiple NLP tasks including:

- **Sentiment Analysis**: Classical methods, BERT, and LSTM models (pre-trained models available).
- **Machine Translation**: English-French translation using Seq2Seq (with pre-trained model and tokenizers), Transformer models, classical methods, and phrase-based pairs.
- **Text Summarization**: Extractive summarization with TextRank and abstractive summarization with T5.
- **Evaluation**: Custom metrics for NLP tasks.

The app is powered by `app.py` and includes prediction scripts and training utilities.

## Project Structure
```
.
├── app.py                  # Flask web application
├── main.py                 # Entry point or additional main script
├── requirements.txt        # Python dependencies
├── predict_translation.py  # Translation prediction script
├── predict_tf.py           # Transformer prediction script
├── static/
│   └── index.html          # Static web assets
├── evaluation/
│   └── metrics.py          # Evaluation metrics
├── sentiment/
│   ├── classical.py
│   ├── bert.py
│   ├── lstm.py
│   └── model.keras         # Pre-trained sentiment model
├── summarization/
│   ├── textrank.py
│   └── t5_summarizer.py
└── translation/
    ├── classical.py
    ├── seq2seq.py
    ├── transformer_mt.py
    ├── transformer_worker.py
    ├── transformer_cache.py
    ├── phrase_pairs.py
    ├── train_seq2seq.py
    ├── saved_seq2seq_model.keras  # Pre-trained Seq2Seq model
    ├── seq2seq_config.json
    ├── eng_tokenizer.json
    └── fr_tokenizer.json
```

## Setup Instructions

1. **Clone/Navigate to the project directory**:
   ```
   cd /Users/preranapradeep/Desktop/6thsem/cie3_NLP
   ```

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
   Key libraries include: torch/tensorFlow, transformers (HuggingFace), nltk, scikit-learn, etc.

3. **(Optional) Install virtual environment** (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your browser to http://127.0.0.1:5050.

The web interface allows interactive testing of sentiment analysis, translation, and summarization tasks.

## Usage Examples

### Sentiment Analysis
- Classical: Rule-based or simple ML.
- BERT: Transformer-based.
- LSTM: Deep learning with pre-trained model.

### Translation (EN ↔ FR)
- Seq2Seq: RNN-based with trained model.
- Transformer: Attention-based.
- Use `predict_translation.py` for CLI predictions.

### Summarization
- TextRank: Extractive.
- T5: Abstractive (HuggingFace).

## Training Models
- Sequence-to-Sequence Translation: `python translation/train_seq2seq.py`
- Other models may have training scripts in respective directories.

## Dependencies
Check `requirements.txt` for full list. Common ones:
- torch / tensorflow
- transformers
- nltk
- scikit-learn
- rouge-score (for evaluation)

## Models
- Pre-trained sentiment model: `sentiment/model.keras`
- Pre-trained Seq2Seq: `translation/saved_seq2seq_model.keras`
- Tokenizers: `eng_tokenizer.json`, `fr_tokenizer.json`

## Contributing
Feel free to extend models, add new NLP tasks, or improve the UI.

## License
MIT License (or specify as needed).

