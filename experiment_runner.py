import json
import os
from os.path import join as path_join
from typing import Callable

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from Common.data_manipulation.dictionary_tools import create_or_append_to_dict_of_lists
from Common.data_manipulation.pandas_tools import save_dataframe
from Common.machine_learning_tools.model_tools.tensoflow_model_tools import initialize_memory_growth
from Common.visualizations.figure_plotting import plot_xy_curves, plot_xs_and_ys
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.utils.generic_utils import Progbar
from keras import optimizers

from custom_callbacks import ReduceLROnPlateauCallback, EarlyStoppingCallback
from losses import SampleWiseCatCrossentropy
from model_builder import build_model


def run_training(run_config, train_generator, val_generator, test_generator, run_path, save_analysis=True, verbose=1):
    print("Starting training...")

    tf.keras.backend.clear_session()
    initialize_memory_growth('GPU')
    do_mixed_precision = initialize_mixed_precision()

    save_run_config(run_config, run_path)

    model = build_model(run_config)
    model.summary()
    # profile = model_profiler(model, 1, use_units=['GPU IDs', 'MFLOPs', 'MB', 'Million', 'MB'], verbose=1)

    optimizer = parse_optimizer(run_config, do_mixed_precision)

    early_stop = False
    callbacks, model_ckpt_fullname, early_stop_cb = build_callbacks(model, optimizer, run_path, run_config['epochs'], verbose)
    loss_obj = SampleWiseCatCrossentropy()

    model.compile(optimizer, loss_obj, run_eagerly=verbose>1,
                  # metrics=[tf.keras.metrics.Accuracy(name='accuracy', threshold=0.9),
                  #          tf.keras.metrics.Precision(name='precision', thresholds=0.9),
                  #          tf.keras.metrics.Recall(name='recall', thresholds=0.9)],
                  )

    #region Step Functions
    @tf.function()
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            losses = loss_obj.call(labels, predictions)
            loss = tf.reduce_mean(losses)
            scaled_loss = optimizer.get_scaled_loss(loss)

        scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
        gradients = optimizer.get_unscaled_gradients(scaled_gradients)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss, losses

    @tf.function()
    def test_step(inputs, labels):
        predictions = model(inputs, training=False)
        losses = loss_obj.call(labels, predictions)
        loss = tf.reduce_mean(losses)
        # if tf.math.is_nan(losses): print("!!! --- loss is nan")
        return loss

    #endregion

    print("Training...")
    history = {}
    sample_losses_log = []

    callbacks.on_train_begin()
    early_stop_cb.on_train_begin()

    for epoch in range(run_config['epochs']):
        # print("Epoch {} / {}".format(epoch, run_config['epochs']))
        progress_bar = Progbar(len(train_generator), stateful_metrics=["epoch"])

        callbacks.on_epoch_begin(epoch)

        train_loss_list = []
        for step_idx in range(len(train_generator)):
            callbacks.on_train_batch_begin(step_idx)

            batch_X, batch_labels = next(train_generator)
            batch_y, sample_indice = batch_labels[:, :-1], batch_labels[:, -1]

            train_loss, train_losses = train_step(batch_X, batch_y)

            if tf.math.is_nan(train_loss):
                print("Loss is nan, stopping training")
                return {'val_acc': 0, 'test_acc': 0}

            progress_bar.update(
                step_idx,
                [("epoch", epoch), ("train_loss", train_loss)]
            )

            batch_losses_log = [[epoch, step_idx, sample_idx, sample_loss]
                                for sample_idx, sample_loss in zip(sample_indice, train_losses.numpy())]
            sample_losses_log.extend(batch_losses_log)

            train_loss_list.append(train_loss)
            callbacks.on_train_batch_end(step_idx+1)

        # Run a validation loop at the end of each epoch.
        mean_val_loss = run_validation(val_generator, test_step)

        history = create_or_append_to_dict_of_lists(history, 'loss', np.mean(train_loss_list))
        history = create_or_append_to_dict_of_lists(history, 'val_loss', mean_val_loss)

        train_generator.on_epoch_end()
        callbacks.on_epoch_end(epoch, {'val_loss': mean_val_loss})
        if early_stop_cb.on_epoch_end(epoch, {'val_loss': mean_val_loss}):
            break

    print()
    print("Done training")
    print()

    save_train_history(history, run_path)
    save_losses_log(run_path, sample_losses_log)

    print("Validating...")
    model.load_weights(model_ckpt_fullname)
    val_acc = evaluate_model(model, val_generator)
    test_acc = evaluate_model(model, test_generator)

    metrics = {'val_acc': val_acc, 'test_acc': test_acc}

    return metrics


def parse_optimizer(run_config, do_mixed_precision):
    optimizer_name = run_config.get('optimizer', 'Adam').lower()

    if optimizer_name == 'adam':
        optimizer = keras.optimizers.Adam(run_config['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    elif optimizer_name == 'gd':
        optimizer = keras.optimizers.SGD(run_config['learning_rate'])
    elif optimizer_name == 'adadelta':
        optimizer = keras.optimizers.Adadelta(run_config['learning_rate'])
    elif optimizer_name == 'adagrad':
        optimizer = keras.optimizers.Adagrad(run_config['learning_rate'])
    elif optimizer_name == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(run_config['learning_rate'])
    elif optimizer_name == 'momentum':
        optimizer = keras.optimizers.SGD(run_config['learning_rate'], momentum=0.9)
    else:
        raise NotImplementedError("Optimizer {} Not Implemented".format(optimizer_name))

    if do_mixed_precision:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer


def save_losses_log(run_path, sample_losses_log):
    training_df = pd.DataFrame(sample_losses_log, columns=['epoch', 'step', 'index', 'loss'])
    training_df_csv_fullname = path_join(run_path, 'training.csv')

    save_dataframe(training_df, training_df_csv_fullname)


def initialize_mixed_precision():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("CUDA enabled: {} | GPU device name: {} | Executing Eagerly: {} | Precision Policy: {}".format(
        tf.test.is_built_with_cuda(), tf.test.gpu_device_name(), tf.executing_eagerly(),
        tf.keras.mixed_precision.global_policy()
    ))
    do_mixed_precision = 'mixed' in tf.keras.mixed_precision.global_policy().name
    return do_mixed_precision


def run_validation(val_generator, test_step: Callable):
    validation_losses = []

    for step_idx in range(len(val_generator)):
        batch_X, batch_y = next(val_generator)

        loss = test_step(batch_X, batch_y)

        if tf.reduce_any(tf.math.is_nan(loss)):
            print("Loss is nan @ step", step_idx)

        validation_losses.append(loss)

    mean_val_loss = tf.reduce_mean(validation_losses).numpy()

    return mean_val_loss


def save_run_config(model_config, run_path):
    # print("model_config")
    # print(model_config)

    json_fullname = path_join(run_path, 'model_config.json')
    with open(json_fullname, "w") as file:
        json.dump(model_config, file)


def build_callbacks(model, optimizer, run_path, epochs=None, verbose=0):
    early_stop = EarlyStoppingCallback(monitor='val_loss', min_delta=0.0001, patience=7, verbose=verbose)

    ckpt_path = path_join(run_path, "checkpoints")
    os.makedirs(ckpt_path, exist_ok=True)
    # model_ckpt_fullname = path_join(run_path, 'cp-{epoch:03d}-{val_binary_accuracy:.3f}-{val_loss:.4f}-{val_precision:.3f}-{val_recall:.3f}.h5')
    model_ckpt_fullname = path_join(ckpt_path, 'cp-best_loss.h5')
    ckpt_callback = ModelCheckpoint(filepath=model_ckpt_fullname, save_best_only=True, save_weights_only=True, verbose=verbose)
    # LR_callback = ReduceLROnPlateau(factor=0.2, patience=4, verbose=verbose)
    LR_callback = ReduceLROnPlateauCallback(factor=0.3, patience=4, min_delta=0.0001, verbose=verbose, optim_lr=optimizer.learning_rate, mode='min')

    callbacks = tf.keras.callbacks.CallbackList(
        [ckpt_callback, LR_callback],
        True, False, model=model, verbose=verbose, epochs=epochs
    )

    return callbacks, model_ckpt_fullname, early_stop


def save_train_history(history, run_path):
    training_loss_plot_save_fullname = path_join(run_path, "train-loss.jpg")
    plot_xs_and_ys(
        [np.arange(len(history['loss'])), np.arange(len(history['val_loss']))],
        [history['loss'], history['val_loss']], ['train loss', 'val loss'],
        marker=False, axis_labels=['Epoch', 'Loss'],
        fig_size=(22, 5), save_fullname=training_loss_plot_save_fullname, only_save=True
    )
    # plot_xy_curves(np.arange(len(history['loss'])), [history['loss'], history['val_loss']],
    #                ['train loss', 'val loss'], axis_labels=['Epoch', 'Loss'],
    #                marker=False, log_scale=False, fig_size=(22, 5), save_fullname=training_loss_plot_save_fullname)

    if "mean_absolute_error" in history:
        training_acc_plot_save_fullname = path_join(run_path, "train-acc.jpg")
        plot_xy_curves(np.arange(len(history['mean_absolute_error'])),
                       [history['mean_absolute_error'], history['val_mean_absolute_error']],
                       ['train mse', 'val mse'], axis_labels=['Epoch', 'Accuracy'],
                       marker=False, log_scale=False, fig_size=(22, 5),
                       save_fullname=training_acc_plot_save_fullname)  # y_limits=[-0.01, 0.6],


def evaluate_model(model, test_generator):
    y_preds = []
    y_true = []
    for step_idx in range(len(test_generator)):
        batch_X, batch_y = next(test_generator)

        y_pred = model.predict(batch_X, verbose=0)

        y_pred = np.argmax(y_pred, axis=1)

        y_preds.append(y_pred)
        y_true.append(batch_y)

    y_preds = np.concatenate(y_preds)
    y_true = np.concatenate(y_true)

    accuracy = accuracy_score(y_true, y_preds)

    return accuracy

    # CED_plot_save_fullname = path_join(run_path, "CED.jpg")

    # error_metrics, CEDs = asd(y_pred, y_test, 1, (22, 5), CED_plot_save_fullname)
    # metrics_df = update_metrics_df(model_name, error_metrics)
    # print()
    # print(''.join(['-'] * 40))
    # print()

    # return error_metrics['mse']



if __name__ == '__main__':
    run_training()