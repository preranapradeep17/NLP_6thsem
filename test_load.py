import tensorflow as tf
from keras.layers import Embedding

class SafeEmbedding(Embedding):
    def __init__(self, **kwargs):
        kwargs.pop("quantization_config", None)
        super().__init__(**kwargs)

try:
    model = tf.keras.models.load_model('translation/saved_seq2seq_model.keras', compile=False, custom_objects={'Embedding': SafeEmbedding})
    print("Success with SafeEmbedding!")
except Exception as e:
    print("Failed with SafeEmbedding:", e)
