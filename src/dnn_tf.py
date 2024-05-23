import keras
import matplotlib.pyplot as plt
from input.func import f1, f2
from utils import create_gif_and_cleanup, data_from_func


class DNN(keras.Model):

    def __init__(self):
        super().__init__()

        self.dense1 = keras.layers.Dense(12, activation='relu')
        self.dense2 = keras.layers.Dense(32, activation='relu')
        self.dense3 = keras.layers.Dense(64, activation='relu')
        self.dense4 = keras.layers.Dense(8, activation='relu')
        self.dense5 = keras.layers.Dense(1, activation='linear')

        self.y_train = None
        self.X_test = None
        self.X_train = None

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
        return super(DNN, self).fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose,
                                    validation_split=validation_split, **kwargs)

    def fit_curve(self, n, x, y):
        for j in range(n):
            self.fit(x=x, y=y, epochs=1, validation_split=0.2)
            self.show_compare(j)

    def predict(self, x, batch_size=None, verbose=0, steps=None, **kwargs):
        return super(DNN, self).predict(x, batch_size=batch_size, verbose=verbose, steps=steps, **kwargs)

    def show_compare(self, it):
        test_df['pred'] = self.predict(self.X_test)
        plt.figure(figsize=(10, 6))
        ax = plt.subplot()
        ax.scatter(self.X_train, self.y_train, color="firebrick", s=2.0)
        ax.scatter(self.X_test, test_df['pred'], color="mediumpurple")
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("y", fontsize=20)

        # CHANGE
        ax.set_xlim([-1, 11])
        ax.set_ylim([-0.1, 2.1])

        ax.tick_params(direction="in", length=10, width=0.8, colors='black')
        ax.spines['top'].set_linewidth(3.0)
        ax.spines['bottom'].set_linewidth(3.0)
        ax.spines['left'].set_linewidth(3.0)
        ax.spines['right'].set_linewidth(3.0)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.savefig('../output/img_' + str(it) + '.png')
        plt.close()

    def save_data(self, X_train, X_test, y_train):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train


if __name__ == "__main__":

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
    model.fit_curve(n=3, x=train_df['X'], y=train_df['y'])

    # create animation and delete png-files
    create_gif_and_cleanup('../output/animation.gif', duration=50)
