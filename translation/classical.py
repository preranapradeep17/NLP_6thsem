class ClassicalTranslator:
    def __init__(self):
        print("🧠 Initialized Rule-Based Translator")

        self.dictionary = {
            "i": "je",
            "am": "suis",
            "you": "tu",
            "hello": "bonjour",
            "world": "monde",
            "love": "aime",
            "hate": "deteste",
            "this": "ce",
            "is": "est",
            "my": "mon",
            "name": "nom"
        }

    def translate(self, text):
        words = text.lower().split()
        return " ".join([self.dictionary.get(w, w) for w in words])