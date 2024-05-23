import keras
import matplotlib.pyplot as plt
from input.func import f1, f2
from src.utils import create_gif_and_cleanup, data_from_func
from src.dnn_tf import DNN

# choose function
f = f2

# load data
train_df, test_df = data_from_func(f)

# plot training data
plt.plot(train_df['X'], train_df['y'])

# create Neural Network
model = DNN()

# save data
model.save_data(train_df['X'], test_df['X'], train_df['y'])

# compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001), loss='mean_squared_error')

# curve fitting
model.fit_curve(n=200, x=train_df['X'], y=train_df['y'])

# create animation and delete png-files
create_gif_and_cleanup('../output/animation.gif', duration=50)
