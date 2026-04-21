import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

try:
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    model = tf.keras.models.load_model("sentiment/saved_dl_model")
    text = sys.argv[1]
    
    pred = float(model.predict([text], verbose=0)[0][0])
    print(pred)
except Exception as e:
    # return a highly confident neutral if it fails or crashes
    print("0.5")
