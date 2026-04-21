import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

warnings.filterwarnings("ignore")

import tensorflow as tf
from datasets import load_dataset

tf.get_logger().setLevel("ERROR")
tf.keras.utils.set_random_seed(42)

try:
    import keras
    KERAS_VERSION = getattr(keras, "__version__", "")
except ImportError:
    KERAS_VERSION = getattr(tf.keras, "__version__", "")

print("⏳ Loading Dataset...")

dataset = load_dataset("glue", "sst2")

train_texts = list(dataset["train"]["sentence"][:20000])
train_labels = list(dataset["train"]["label"][:20000])
val_texts = list(dataset["validation"]["sentence"])
val_labels = list(dataset["validation"]["label"])

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
    .shuffle(len(train_texts), seed=42)
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((val_texts, val_labels))
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=20000,
    output_sequence_length=120
)

vectorizer.adapt(train_texts)

model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(20000, 128, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("🔥 Training Model...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True,
    )
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    callbacks=callbacks,
)

print("✅ Saving Model...")

if KERAS_VERSION.startswith("3."):
    model.save("sentiment/model.keras")

saved_model_path = "sentiment/saved_dl_model"

try:
    if hasattr(model, "export"):
        model.export(saved_model_path)
    else:
        model.save(saved_model_path, save_format="tf")
except Exception as exc:
    print(f"⚠️ SavedModel export skipped: {exc}")
