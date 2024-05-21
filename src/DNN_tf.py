import os
import math
import keras
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def get_model():
    model_input = keras.Input(shape=(1,))
    x = keras.layers.Dense(12, activation='relu')(model_input)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    x = keras.layers.Dense(1, activation='linear')(x)
    model = keras.Model(model_input, x)
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(learning_rate=0.00001))
    return model


def show_compare(it):
    test_df['pred'] = model.predict(test_df['X'])
    plt.figure(figsize=(12, 6))
    ax = plt.subplot()
    ax.scatter(train_df['X'], train_df['y'], color="red")
    ax.scatter(test_df['X'], test_df['pred'], color="purple")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_xlim([-1, 10])
    ax.set_ylim([-0.1, 2.1])
    ax.tick_params(direction="in", length=10, width=0.8, colors='black')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    plt.savefig('../output/img_' + str(it) + '.png')
    plt.close()


def create_gif_and_cleanup(output_path, duration=500):
    # Get list of all image files in the output directory
    png_files = [f for f in os.listdir('../output') if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort files numerically

    # List to hold the images
    images = []

    # Open each file and add it to the images list
    for file_name in png_files:
        file_path = os.path.join('../output', file_name)
        images.append(Image.open(file_path))

    # Save images as a GIF
    if images:
        images[0].save(output_path, save_all=True, append_images=images[1:], optimize=False, duration=duration, loop=0)

    # Delete the PNG files
    for file_name in png_files:
        file_path = os.path.join('../output', file_name)
        os.remove(file_path)


if __name__ == "__main__":

    # training data
    train_df = pd.DataFrame(np.random.sample([100000, 1]) * 10, columns=['X'])
    train_df['y'] = train_df['X'].apply(lambda x: math.sin(x) + 1)
    train_df = train_df.sample(len(train_df))

    # test data
    test_df = pd.DataFrame(np.random.sample([100, 1]) * 10, columns=['X'])

    # plot training data
    plt.scatter(train_df['X'], train_df['y'])

    # fit curve with DNN
    model = get_model()
    for i in range(0, 100):
        history = model.fit(train_df['X'], train_df['y'], epochs=1, validation_split=0.2)
        show_compare(i)

    # create animation and delete png-files
    create_gif_and_cleanup('../output/animation.gif', duration=50)
