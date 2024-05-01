import math

import keras
import numpy as np
import tensorflow as tf


class SimpleGenerator(tf.compat.v2.keras.utils.Sequence):

    def __init__(self, X_data, y_data, sample_indice=None, batch_size=128, n_classes=10, gen_y=True, shuffle=True, preprocess=True):
        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        self.sample_indice = sample_indice
        self.gen_y = gen_y
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.preprocess = preprocess

        self.n = 0
        self.list_IDs = np.arange(len(self.X_data))
        self.on_epoch_end()

    def __next__(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end()
            self.n = 0

        return data

    def __len__(self):
        # Return the number of batches of the dataset
        return math.ceil(len(self.indexes) / self.batch_size)

    def __getitem__(self, index):
        # Generate indexes of the batch
        # import pdb;pdb.set_trace()
        indexes = self.indexes[index * self.batch_size:
                               (index + 1) * self.batch_size]  # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self._generate_x(list_IDs_temp)

        if self.gen_y:
            y = self._generate_y(list_IDs_temp)
            return X, y

        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X_data))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_x(self, list_IDs_temp):
        # X = np.empty((self.batch_size, *self.dim))
        # for i, ID in enumerate(list_IDs_temp):
        #     X[i,] = self.X_data[ID]

        X = np.array([self.X_data[ID] for ID in list_IDs_temp])

        # Normalize data
        X = X.astype('float32')

        if self.preprocess:
            X /= 255.0
            X = X[:, :, :, np.newaxis]

        return X

    def _generate_y(self, list_IDs_temp):
        # y = np.empty(self.batch_size)
        # for i, ID in enumerate(list_IDs_temp):
        #     y[i] = self.y_data[ID]

        y = np.array([self.y_data[ID] for ID in list_IDs_temp])

        if self.preprocess:
            y = keras.utils.to_categorical(y, num_classes=self.n_classes)

        if self.sample_indice is not None:
            y_indice = np.array([self.sample_indice[ID] for ID in list_IDs_temp])

            # print(y.shape, y_indice[:, np.newaxis].shape)
            y = np.hstack([y[:, np.newaxis], y_indice[:, np.newaxis]])


        return y