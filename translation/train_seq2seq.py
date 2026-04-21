import argparse
import json
import os
import re
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

warnings.filterwarnings("ignore")

import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from translation.phrase_pairs import COMMON_PHRASE_PAIRS
from translation.seq2seq import normalize_translation_text

tf.get_logger().setLevel("ERROR")
tf.keras.utils.set_random_seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["opus_books", "opus100"], default="opus_books")
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--src-vocab-size", type=int, default=8000)
    parser.add_argument("--tgt-vocab-size", type=int, default=10000)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--encoder-max-len", type=int, default=12)
    parser.add_argument("--decoder-max-len", type=int, default=14)
    parser.add_argument("--shuffle-buffer", type=int, default=50000)
    parser.add_argument("--seed-repeats", type=int, default=14)
    return parser.parse_args()


def is_clean_sentence_pair(english, french, encoder_max_len, decoder_max_len):
    source_tokens = english.split()
    target_tokens = french.split()

    if len(source_tokens) < 2 or len(target_tokens) < 1:
        return False

    if len(source_tokens) > encoder_max_len:
        return False

    if len(target_tokens) > decoder_max_len - 2:
        return False

    pair_text = f"{english} {french}"
    if any(char.isdigit() for char in pair_text):
        return False

    if re.search(r"[\[\]{}<>@=_/\\|]", pair_text):
        return False

    if english.isupper() or french.isupper():
        return False

    return True


def iter_dataset_rows(dataset_name, shuffle_buffer):
    if dataset_name == "opus_books":
        print("⏳ Loading OPUS Books en-fr examples...")
        dataset = load_dataset("opus_books", "en-fr", split="train").shuffle(seed=42)
        for row in dataset:
            yield row
        return

    print("⏳ Streaming OPUS-100 en-fr examples...")
    dataset = load_dataset(
        "Helsinki-NLP/opus-100",
        "en-fr",
        split="train",
        streaming=True,
    ).shuffle(seed=42, buffer_size=shuffle_buffer)
    for row in dataset:
        yield row


def collect_examples(dataset_name, limit, encoder_max_len, decoder_max_len, shuffle_buffer):
    dataset_rows = iter_dataset_rows(dataset_name, shuffle_buffer)

    source_texts = []
    target_texts = []

    for row in dataset_rows:
        english = normalize_translation_text(row["translation"]["en"])
        french = normalize_translation_text(row["translation"]["fr"])

        if not english or not french:
            continue

        if not is_clean_sentence_pair(english, french, encoder_max_len, decoder_max_len):
            continue

        source_texts.append(english)
        target_texts.append(f"startseq {french} endseq")

        if len(source_texts) >= limit:
            break

    if len(source_texts) < limit:
        raise RuntimeError(f"Only collected {len(source_texts)} examples out of requested {limit}.")

    return source_texts, target_texts


def add_seed_pairs(train_source, train_target, repeats):
    augmented_source = list(train_source)
    augmented_target = list(train_target)

    for _ in range(repeats):
        for english, french in COMMON_PHRASE_PAIRS:
            augmented_source.append(normalize_translation_text(english))
            augmented_target.append(
                f"startseq {normalize_translation_text(french)} endseq"
            )

    return augmented_source, augmented_target


def build_tokenizer(texts, vocab_size):
    tokenizer = Tokenizer(
        num_words=vocab_size,
        filters="",
        lower=False,
        oov_token="<unk>",
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_artifacts(model, eng_tokenizer, fr_tokenizer, config):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "saved_seq2seq_model.keras")
    eng_tokenizer_path = os.path.join(base_dir, "eng_tokenizer.json")
    fr_tokenizer_path = os.path.join(base_dir, "fr_tokenizer.json")
    config_path = os.path.join(base_dir, "seq2seq_config.json")

    model.save(model_path)

    with open(eng_tokenizer_path, "w", encoding="utf-8") as f:
        f.write(eng_tokenizer.to_json())

    with open(fr_tokenizer_path, "w", encoding="utf-8") as f:
        f.write(fr_tokenizer.to_json())

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def main():
    args = parse_args()
    total_examples = args.train_size + args.val_size

    source_texts, target_texts = collect_examples(
        args.dataset,
        total_examples,
        args.encoder_max_len,
        args.decoder_max_len,
        args.shuffle_buffer,
    )
    train_source = source_texts[: args.train_size]
    val_source = source_texts[args.train_size :]
    train_target = target_texts[: args.train_size]
    val_target = target_texts[args.train_size :]
    train_source, train_target = add_seed_pairs(
        train_source,
        train_target,
        args.seed_repeats,
    )

    print("🧠 Building tokenizers...")
    eng_tokenizer = build_tokenizer(train_source, args.src_vocab_size)
    fr_tokenizer = build_tokenizer(train_target, args.tgt_vocab_size)

    train_encoder = eng_tokenizer.texts_to_sequences(train_source)
    val_encoder = eng_tokenizer.texts_to_sequences(val_source)
    train_decoder_full = fr_tokenizer.texts_to_sequences(train_target)
    val_decoder_full = fr_tokenizer.texts_to_sequences(val_target)

    encoder_train = pad_sequences(
        train_encoder,
        maxlen=args.encoder_max_len,
        padding="post",
        truncating="post",
    )
    encoder_val = pad_sequences(
        val_encoder,
        maxlen=args.encoder_max_len,
        padding="post",
        truncating="post",
    )

    decoder_train_full = pad_sequences(
        train_decoder_full,
        maxlen=args.decoder_max_len,
        padding="post",
        truncating="post",
    )
    decoder_val_full = pad_sequences(
        val_decoder_full,
        maxlen=args.decoder_max_len,
        padding="post",
        truncating="post",
    )

    decoder_train_input = decoder_train_full[:, :-1]
    decoder_train_target = decoder_train_full[:, 1:]
    decoder_val_input = decoder_val_full[:, :-1]
    decoder_val_target = decoder_val_full[:, 1:]
    train_sample_weights = (decoder_train_target != 0).astype("float32")
    val_sample_weights = (decoder_val_target != 0).astype("float32")

    src_vocab_size = min(args.src_vocab_size, len(eng_tokenizer.word_index) + 1)
    tgt_vocab_size = min(args.tgt_vocab_size, len(fr_tokenizer.word_index) + 1)

    print("🔥 Building seq2seq model...")
    encoder_inputs = tf.keras.Input(shape=(args.encoder_max_len,), name="encoder_inputs")
    encoder_embedding = tf.keras.layers.Embedding(
        src_vocab_size,
        args.embedding_dim,
        mask_zero=True,
        name="encoder_embedding",
    )(encoder_inputs)
    encoder_outputs, encoder_state_h, encoder_state_c = tf.keras.layers.LSTM(
        args.latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=0.2,
        name="encoder_lstm",
    )(encoder_embedding)

    decoder_inputs = tf.keras.Input(shape=(args.decoder_max_len - 1,), name="decoder_inputs")
    decoder_embedding = tf.keras.layers.Embedding(
        tgt_vocab_size,
        args.embedding_dim,
        mask_zero=True,
        name="decoder_embedding",
    )(decoder_inputs)
    decoder_outputs, _, _ = tf.keras.layers.LSTM(
        args.latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=0.2,
        name="decoder_lstm",
    )(decoder_embedding, initial_state=[encoder_state_h, encoder_state_c])

    attention_context = tf.keras.layers.AdditiveAttention(
        name="decoder_attention"
    )([decoder_outputs, encoder_outputs])
    decoder_outputs = tf.keras.layers.Concatenate(name="decoder_context_concat")(
        [decoder_outputs, attention_context]
    )
    decoder_outputs = tf.keras.layers.Dense(
        args.latent_dim,
        activation="tanh",
        name="decoder_projection",
    )(decoder_outputs)
    decoder_outputs = tf.keras.layers.Dense(
        tgt_vocab_size,
        activation="softmax",
        name="decoder_classifier",
    )(decoder_outputs)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                (encoder_train, decoder_train_input),
                decoder_train_target,
                train_sample_weights,
            )
        )
        .shuffle(len(encoder_train), seed=42)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                (encoder_val, decoder_val_input),
                decoder_val_target,
                val_sample_weights,
            )
        )
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        )
    ]

    print("🚀 Training seq2seq model...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    config = {
        "encoder_max_len": args.encoder_max_len,
        "decoder_max_len": args.decoder_max_len,
        "start_token": fr_tokenizer.word_index["startseq"],
        "end_token": fr_tokenizer.word_index["endseq"],
    }

    print("✅ Saving seq2seq artifacts...")
    save_artifacts(model, eng_tokenizer, fr_tokenizer, config)


if __name__ == "__main__":
    main()
