from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class AbstractiveSummarizer:
    def __init__(self):
        print("⏳ Loading T5 Transformer Model...")
        self.model_name = "t5-small"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        print("✅ T5 Model Loaded!")

    def summarize(self, text):
        """
        Generate abstractive summary using T5
        """
        input_text = "summarize: " + text

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=50,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


# 🔍 Test
if __name__ == "__main__":
    sample_text = """
    Artificial intelligence is intelligence demonstrated by machines.
    It is widely used in modern applications such as search engines, recommendation systems, and autonomous vehicles.
    AI systems can process large amounts of data and improve over time.
    """

    model = AbstractiveSummarizer()

    print("\n📝 Abstractive Summary:")
    print(model.summarize(sample_text))