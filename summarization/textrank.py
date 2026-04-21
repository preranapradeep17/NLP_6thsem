import re

import nltk

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


class ExtractiveSummarizer:
    def __init__(self):
        print("🧠 Initialized TextRank Extractive Summarizer")
        self.summarizer = TextRankSummarizer()
        self.has_punkt = self._has_punkt_tokenizer()

    def _has_punkt_tokenizer(self):
        for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
            try:
                nltk.data.find(resource)
                return True
            except LookupError:
                continue
        return False

    def _fallback_summarize(self, text, sentences_count):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]
        return " ".join(sentences[:sentences_count])

    def summarize(self, text, sentences_count=2):
        """
        Extract important sentences from text
        """
        if not self.has_punkt:
            return self._fallback_summarize(text, sentences_count)

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = self.summarizer(parser.document, sentences_count)

        return " ".join(str(sentence) for sentence in summary)


# 🔍 Test
if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is transforming industries by enabling machines to learn from data.
    It is used in healthcare, finance, education, and transportation.
    AI helps automate repetitive tasks and improves decision-making.
    However, ethical concerns such as bias and privacy still remain important challenges.
    """

    model = ExtractiveSummarizer()

    print("\n📄 Extractive Summary:")
    print(model.summarize(sample_text))
