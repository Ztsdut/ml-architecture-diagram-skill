from tensorflow import keras
from tensorflow.keras import layers

image = keras.Input(shape=(128, 128, 3), name="image")
metadata = keras.Input(shape=(16,), name="metadata")
x = layers.Conv2D(32, 3, activation="relu")(image)
x = layers.GlobalAveragePooling2D()(x)
m = layers.Dense(32, activation="relu")(metadata)
fused = layers.Concatenate()([x, m])
out = layers.Dense(5)(fused)
model = keras.Model(inputs=[image, metadata], outputs=out)
