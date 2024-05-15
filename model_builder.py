import keras
import tensorflow as tf
from keras import Sequential
from keras.engine.input_layer import InputLayer
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras import activations, initializers, regularizers


BATCH_NORM_MOMENTUM = 0.997
BATCH_NORM_EPSILON = 1e-5


def build_model(run_config):
    architecture = run_config['architecture']
    if architecture == 'lenet5':
        return build_lenet5(run_config)

    elif architecture == 'LeNet1':
        return build_LeNet1_model(run_config)

    elif architecture == 'LeNet2':
        return build_LeNet2_model(run_config)

    elif architecture == 'LeNet':
        return build_LeNet_model(run_config)

    else:
        raise NotImplementedError("Architecture '{}' not implemented".format(architecture))


def build_LeNet1_model(model_config):
    tf.keras.backend.set_image_data_format(model_config['data_format'])

    model = Sequential()
    model.add(InputLayer(input_shape=model_config['input_shape']))

    model.add(Conv2D(int(model_config['conv1_depth']),
                     kernel_size=(int(model_config['window_size']), int(model_config['window_size'])),
                     activation=model_config['activation'], padding='same'))
    model.add(MaxPooling2D((int(model_config['pool1_size']), int(model_config['pool1_size'])), padding='same'))

    model.add(Conv2D(int(model_config['conv2_depth']),
                     kernel_size=(int(model_config['window_size']), int(model_config['window_size'])),
                     activation=model_config['activation'],
                     padding='same'))
    model.add(MaxPooling2D((int(model_config['pool2_size']), int(model_config['pool2_size'])), padding='same'))

    model.add(Dropout(float(1.0 - model_config['dropout_rate'])))
    model.add(Flatten())

    model.add(Dense(int(model_config['fc_depth']), activation='relu',
                    kernel_regularizer=keras.regularizers.l2(model_config['reg_param'])))
    model.add(Dropout(float(1.0 - model_config['dropout_rate'])))
    model.add(Dense(model_config['num_classes'], activation='softmax', dtype='float32'))

    return model


def build_LeNet2_model(model_config):
    tf.keras.backend.set_image_data_format(model_config['data_format'])
    window_shape = (int(model_config['window_size']), int(model_config['window_size']))

    model = Sequential()
    model.add(InputLayer(input_shape=model_config['input_shape']))

    model.add(Conv2D(int(model_config['conv1_depth']), kernel_size=window_shape,
                     kernel_initializer=keras.initializers.HeNormal(),
                     kernel_regularizer=keras.regularizers.l2(model_config['reg_param']),
                     padding='same'))
    if bool(model_config.get('batch_normalization', 0)):
        model.add(BatchNormalization())
    model.add(parse_activation(model_config['activation']))

    model.add(MaxPooling2D((int(model_config['pool1_size']), int(model_config['pool1_size'])), padding='same'))

    model.add(Conv2D(int(model_config['conv2_depth']), kernel_size=window_shape,
                     kernel_regularizer=keras.regularizers.l2(model_config['reg_param']),
                     kernel_initializer=keras.initializers.HeNormal(), padding='same'))
    if model_config.get('batch_normalization', 0):
        model.add(BatchNormalization())
    model.add(parse_activation(model_config['activation']))

    model.add(MaxPooling2D((int(model_config['pool2_size']), int(model_config['pool2_size'])), padding='same'))

    model.add(Flatten())

    model.add(Dense(int(model_config['fc_depth']),
                    kernel_initializer=keras.initializers.HeNormal(),
                    kernel_regularizer=keras.regularizers.l2(model_config['reg_param'])))
    if model_config.get('batch_normalization', 0):
        model.add(BatchNormalization())
    model.add(parse_activation(model_config['activation']))

    model.add(Dropout(model_config['dropout_rate']))
    model.add(Dense(model_config['num_classes'], activation='softmax', dtype='float32'))

    return model


def build_LeNet_model(model_config):
    tf.keras.backend.set_image_data_format(model_config['data_format'])

    model = Sequential()
    model.add(InputLayer(input_shape=model_config['input_shape']))

    for i in range(model_config.get("n_conv", 0)):
        layer_idx = i + 1

        conv_depth = model_config.get("conv_depth", False) or model_config.get("conv%s_depth" % layer_idx, False)
        window_size = model_config.get('window_size', False) or model_config.get("window%s_size" % layer_idx, False)
        l2_reg = model_config.get("l2_reg", False)

        model.add(Conv2D(int(conv_depth), kernel_size=(int(window_size), int(window_size)),
                         kernel_regularizer=keras.regularizers.l2(l2_reg),
                         kernel_initializer=keras.initializers.HeNormal(), padding='same'))

        regularization = model_config.get('reg_method', False)
        pool_size = model_config.get("pool_size", False) or model_config.get("pool%s_size" % layer_idx, False)

        if regularization and regularization == "BatchNorm":
            model.add(BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON))

        if model_config.get('activation', False):
            model.add(parse_activation(model_config['activation']))

        if regularization and regularization == "Dropout":
            model.add(Dropout(model_config['dropout_rate']))

        if pool_size:
            model.add(MaxPooling2D((int(pool_size), int(pool_size)), padding='same'))

    model.add(Flatten())

    for layer_idx in range(model_config.get("n_deep", 0)):

        fc_depth = model_config.get('fc_depth', False) or model_config.get("fc%s_depth" % layer_idx, False)
        l2_reg = model_config.get("l2_reg", False)

        model.add(Dense(int(fc_depth),
                        kernel_initializer=keras.initializers.HeNormal(),
                        kernel_regularizer=keras.regularizers.l2(l2_reg)))

        regularization = model_config.get('reg_method', False)
        if regularization and regularization == "BatchNorm":
            model.add(BatchNormalization(momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON))

        if model_config.get('activation', False):
            model.add(parse_activation(model_config['activation']))

        if regularization and regularization == "Dropout":
            model.add(Dropout(model_config['dropout_rate']))

    model.add(Dense(model_config['num_classes'], activation='softmax', dtype='float32'))

    return model


def build_lenet5(config):
    model = Sequential()

    # build stem

    for i in config['n_layers']:
        model.add(Conv2D(int(config['layer_depth']), kernel_size=int(config['window_size']),
                         activation=config['act_f'],
                         input_shape=config["input_shape"],
                         padding='same')
        )

        model.add(MaxPooling2D(pool_size=int(config['p1_size']), padding='same'))

        if config['name'] == 'dropout_rate':
            model.add(Dropout(float(config['dropout_rate'])))

    if "dense_blocks" in config:
        dense_blocks_config = config['dense_blocks']
        model.add(Flatten())

        for i in dense_blocks_config['n_layers']:
            model.add(Dense(int(dense_blocks_config['depth']), activation=dense_blocks_config['act_f'],
                            kernel_regularizer=keras.regularizers.L1(dense_blocks_config['reg_param'])))

            model.add(BatchNormalization())

            if dense_blocks_config['name'] == 'dropout_rate':
                model.add(Dropout(float(dense_blocks_config['dropout_rate'])))

    model.add(Dense(config["output_size"], activation=config["output_f"]))

    return model


def parse_activation(activation_name):
    if activation_name == 'relu':
        act = keras.layers.Activation(activations.relu)
    elif activation_name == 'tanh':
        act = keras.layers.Activation(activations.tanh)
    elif activation_name == 'sigmoid':
        act = keras.layers.Activation(activations.sigmoid)
    elif activation_name == 'elu':
        act = keras.layers.Activation(activations.elu)
    elif activation_name == 'lrelu':
        act = keras.layers.LeakyReLU(0.2)
    else:
        raise NotImplementedError("Activation {} Not implemented".format(activation_name))

    return act


# def build_flexible_model(model_config, input_size=1280, output_size=3):
#     layers_list = [tf.keras.layers.InputLayer(input_shape=(input_size))]
#
#     for arch_row in model_config['arch']:
#         layers_list.append(tf.keras.layers.Dense(arch_row['units']))
#
#         normalisation = arch_row.get('norm', False)
#         if normalisation == 'layer':
#             layers_list.append(tf.keras.layers.LayerNormalization())
#         elif normalisation == 'batch':
#             layers_list.append(tf.keras.layers.BatchNormalization())
#
#         if arch_row.get('dropout', 0):
#             layers_list.append(tf.keras.layers.Dropout(arch_row['dropout']))
#
#         layers_list.append(tf.keras.layers.Activation(arch_row['act']))
#
#     layers_list.append(tf.keras.layers.Dense(output_size))
#
#     model = tf.keras.Sequential(layers_list)
#
#     return model