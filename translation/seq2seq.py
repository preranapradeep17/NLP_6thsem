import json
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from translation.phrase_pairs import COMMON_PHRASE_PAIRS


def normalize_translation_text(text):
    text = text.lower().strip()
    text = re.sub(r"([?.!,;:'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detokenize_translation_text(text):
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


class DLTranslator:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(base_dir, "saved_seq2seq_model.keras")
        self.eng_tokenizer_path = os.path.join(base_dir, "eng_tokenizer.json")
        self.fr_tokenizer_path = os.path.join(base_dir, "fr_tokenizer.json")
        self.config_path = os.path.join(base_dir, "seq2seq_config.json")
        self.is_trained = False
        self.identity_blocklist = {
            "happy",
            "sad",
            "tired",
            "sorry",
            "hungry",
            "angry",
            "scared",
            "fine",
            "okay",
            "ok",
            "well",
            "ready",
            "busy",
            "late",
            "early",
            "sick",
            "lost",
            "here",
            "there",
            "alone",
        }
        self.seed_lookup = {
            normalize_translation_text(english): french
            for english, french in COMMON_PHRASE_PAIRS
        }

        print("⏳ Checking DL Translation Model...")

        if self._artifacts_exist():
            self._load_artifacts()
            self.is_trained = True
            print("✅ DL Translation Model Loaded!")
        else:
            print("⚠️ DL Translation Model Not Trained.")

    def _artifacts_exist(self):
        return all(
            os.path.exists(path)
            for path in (
                self.model_path,
                self.eng_tokenizer_path,
                self.fr_tokenizer_path,
                self.config_path,
            )
        )

    def _load_artifacts(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

        with open(self.eng_tokenizer_path, "r", encoding="utf-8") as f:
            self.eng_tokenizer = tokenizer_from_json(f.read())

        with open(self.fr_tokenizer_path, "r", encoding="utf-8") as f:
            self.fr_tokenizer = tokenizer_from_json(f.read())

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.encoder_max_len = config["encoder_max_len"]
        self.decoder_max_len = config["decoder_max_len"]
        self.start_token = config["start_token"]
        self.end_token = config["end_token"]

    def _fallback_translation(self, text):
        normalized = normalize_translation_text(text)
        if normalized in self.seed_lookup:
            return self.seed_lookup[normalized]

        name_match = re.fullmatch(r"my name is ([a-z][a-z '\-]*)", normalized)
        if name_match:
            name = " ".join(part.capitalize() for part in name_match.group(1).split())
            return f"Je m'appelle {name}."

        identity_match = re.fullmatch(r"i am ([a-z][a-z '\-]*)", normalized)
        if identity_match:
            candidate_words = identity_match.group(1).split()
            if (
                1 <= len(candidate_words) <= 3
                and all(word not in self.identity_blocklist for word in candidate_words)
            ):
                name = " ".join(part.capitalize() for part in candidate_words)
                return f"Je suis {name}."

        return None

    def _looks_unstable(self, text):
        normalized = normalize_translation_text(text)
        words = normalized.split()
        if not words:
            return True

        if "<unk>" in words or words.count("-") >= 2:
            return True

        if len(words) >= 6 and len(set(words)) <= max(2, len(words) // 3):
            return True

        repeated_run = 1
        for idx in range(1, len(words)):
            if words[idx] == words[idx - 1]:
                repeated_run += 1
                if repeated_run >= 3:
                    return True
            else:
                repeated_run = 1

        return False

    def translate(self, text):
        fallback = self._fallback_translation(text)
        if fallback:
            return fallback

        if not self.is_trained:
            return "[DL Seq2Seq model not trained yet. Train and save a translation model to enable this output.]"

        normalized = normalize_translation_text(text)
        encoder_seq = self.eng_tokenizer.texts_to_sequences([normalized])
        encoder_input = pad_sequences(
            encoder_seq,
            maxlen=self.encoder_max_len,
            padding="post",
        )

        decoded_ids = [self.start_token]

        for _ in range(self.decoder_max_len - 1):
            decoder_input = pad_sequences(
                [decoded_ids],
                maxlen=self.decoder_max_len - 1,
                padding="post",
            )

            pred = self.model.predict(
                [tf.constant(encoder_input), tf.constant(decoder_input)],
                verbose=0,
            )

            step_index = len(decoded_ids) - 1
            next_id = int(np.argmax(pred[0, step_index]))

            if next_id == 0 or next_id == self.end_token:
                break

            decoded_ids.append(next_id)

        words = [
            self.fr_tokenizer.index_word[token_id]
            for token_id in decoded_ids[1:]
            if token_id in self.fr_tokenizer.index_word
        ]

        words = [word for word in words if word not in {"startseq", "endseq"}]

        if not words:
            return "[DL Seq2Seq model produced no translation.]"

        translated = detokenize_translation_text(" ".join(words))
        if self._looks_unstable(translated):
            return "[DL Seq2Seq translation uncertain for this sentence.]"

        return translated
