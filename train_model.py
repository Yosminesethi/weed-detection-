import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# load dataset
data = np.load("Data.npz")

x_train = data['arr_0']
x_test = data['arr_1']
y_train = data['arr_2']
y_test = data['arr_3']

# normalize images (0–255 → 0–1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# build model
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(70,70,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(12, activation='softmax')  # 12 classes
])

# compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# save model
model.save("weed_model.h5")

print("Model training complete!")