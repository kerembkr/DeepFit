import os
import math
import keras
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


class DNN(keras.Model):

    def __init__(self):
        super().__init__()

        self.dense1 = keras.layers.Dense(12, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.dense3 = keras.layers.Dense(64, activation='relu')
        self.dense4 = keras.layers.Dense(8, activation='relu')
        self.dense5 = keras.layers.Dense(1, activation='linear')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        return x

    def compile(self, optimizer, loss, metrics=None, **kwargs):
        super(DNN, self).compile(optimizer=optimizer, loss=loss, metrics=metrics, **kwargs)

    def fit(self, x, y, batch_size=None, epochs=1, verbose=1, validation_split=0.0, **kwargs):
        return super(DNN, self).fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, **kwargs)

    def predict(self, x, batch_size=None, verbose=0, steps=None, **kwargs):
        return super(DNN, self).predict(x, batch_size=batch_size, verbose=verbose, steps=steps, **kwargs)


if __name__ == "__main__":
    model = DNN()

    # Dummy data for demonstration
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)

    # Compile the model
    model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    # Fit the model
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=10,
        epochs=5,
        verbose=1,
        validation_split=0.2
    )

    # Predict with the model
    x_test = np.random.rand(10, 10)
    predictions = model.predict(x_test)
    print(predictions)