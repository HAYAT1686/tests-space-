import tensorflow as tf

def create_simple_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(32,40)),
        tf.keras.layers.Dense(1)
    ])
    return model
