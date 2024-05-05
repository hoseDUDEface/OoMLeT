import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


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

            model.add(keras.layers.normalization.BatchNormalization())

            if dense_blocks_config['name'] == 'dropout_rate':
                model.add(Dropout(float(dense_blocks_config['dropout_rate'])))


    model.add(Dense(config["output_size"], activation=config["output_f"]))

    return model
