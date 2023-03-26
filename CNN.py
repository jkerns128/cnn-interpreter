import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib


from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

CNN = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

CNN.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#TODO Try out cifar100
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = np.concatenate((x_train, x_test))
y_train = np.concatenate((y_train, y_test))

x_train = x_train / 255.0
y_train = y_train.reshape(-1,)

CNN.fit(x_train, y_train, epochs=10)

CNN.save(pathlib.Path(__file__).parent.resolve())
