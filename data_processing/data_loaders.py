import numpy as np
from keras.datasets import mnist, cifar10, cifar100


def load_defined_data(dataset_name: str, data_format='channels_first', do_preprocess=True):
    if dataset_name.upper() == "MNIST":
        dataset = get_mnist_data(data_format, do_preprocess)

    elif dataset_name.upper() == "CIFAR-10":
        dataset = get_cifar10_data(data_format, do_preprocess)

    elif dataset_name.upper() == "CIFAR-100":
        dataset = get_cifar100_data(data_format, do_preprocess)

    else:
        raise NotImplementedError("Dataset {} is not implemented".format(dataset_name))

    return dataset


def get_mnist_data(data_format: str = 'channels_first', do_preprocess: bool = True):
    img_rows = 28
    img_cols = 28
    img_channels = 1
    num_classes = 10
    n_train = 55000
    n_valid = 5000

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    dataset_tuple = mnist.load_data()
    dataset = format_keras_dataset(dataset_tuple, data_format, do_preprocess, img_channels, img_cols, img_rows, n_train, n_valid, num_classes)

    return dataset


def get_cifar10_data(data_format: str = 'channels_first', do_preprocess: bool = True):
    img_rows = 32
    img_cols = 32
    img_channels = 3
    num_classes = 10
    n_train = 45000
    n_valid = 5000

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    dataset_tuple = cifar10.load_data()
    dataset = format_keras_dataset(dataset_tuple, data_format, do_preprocess, img_channels, img_cols, img_rows, n_train, n_valid, num_classes)

    return dataset


def get_cifar100_data(data_format: str = 'channels_first', do_preprocess: bool = True):
    img_rows = 32
    img_cols = 32
    img_channels = 3
    num_classes = 100
    n_train = 45000
    n_valid = 5000

    # the data, split between train and test sets
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    dataset_tuple = cifar100.load_data()
    dataset = format_keras_dataset(dataset_tuple, data_format, do_preprocess, img_channels, img_cols, img_rows, n_train, n_valid, num_classes)

    return dataset


def format_keras_dataset(dataset_tuple, data_format, do_preprocess, img_channels, img_cols, img_rows, n_train, n_valid, num_classes):
    (x_train, y_train), (x_test, y_test) = dataset_tuple

    if data_format == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
        input_shape = (img_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
        input_shape = (img_rows, img_cols, img_channels)

    if do_preprocess:
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # zero-one normalization
        x_train /= 255
        x_test /= 255

    # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    x_train, y_train = x_train[:n_train], np.squeeze(y_train[:n_train])
    x_valid, y_valid = x_train[-n_valid:], np.squeeze(y_train[-n_valid:])
    y_test = np.squeeze(y_test)

    dataset = {'num_classes': num_classes,
               'x_train': x_train, 'y_train': y_train,
               'x_val': x_valid, 'y_val': y_valid,
               'x_test': x_test, 'y_test': y_test,
               'input_shape': input_shape}

    return dataset