import os
from PIL import Image
import pandas as pd
import numpy as np


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


def data_from_func(f):

    # training data
    train_df = pd.DataFrame(np.random.sample([100000, 1]) * 10, columns=['X'])
    train_df['y'] = train_df['X'].apply(lambda x: f(x))
    train_df = train_df.sample(len(train_df))

    # testing data
    test_df = pd.DataFrame(np.random.sample([100, 1]) * 10, columns=['X'])

    return train_df, test_df
