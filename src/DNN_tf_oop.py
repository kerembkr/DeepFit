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


if __name__ == "__main__":
    model = DNN()
