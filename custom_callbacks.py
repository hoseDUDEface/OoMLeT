import numpy as np
import tensorflow as tf
from keras.callbacks import Callback


class ReduceLROnPlateauCallback(Callback):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This callback monitors a
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.
    Example:
    ```python
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.001)
    model.fit(X_train, Y_train, callbacks=[reduce_lr])
    ```
    Arguments:
        monitor: quantity to be monitored.
        factor: factor by which the learning rate will be reduced. new_lr = lr *
          factor
        patience: number of epochs with no improvement after which learning rate
          will be reduced.
        verbose: int. 0: quiet, 1: update messages.
        mode: one of {auto, min, max}. In `min` mode, lr will be reduced when the
          quantity monitored has stopped decreasing; in `max` mode it will be
          reduced when the quantity monitored has stopped increasing; in `auto`
          mode, the direction is automatically inferred from the name of the
          monitored quantity.
        min_delta: threshold for measuring the new optimum, to only focus on
          significant changes.
        cooldown: number of epochs to wait before resuming normal operation after
          lr has been reduced.
        min_lr: lower bound on the learning rate.
    """

    def __init__(self,
                 ## Custom modification:  Deprecated due to focusing on validation loss
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 mode='auto',
                 min_delta=1e-4,
                 cooldown=0,
                 min_lr=0,
                 sign_number=4,
                 ## Custom modification: Passing optimizer as arguement
                 optim_lr=None,
                 ## Custom modification:  linearly reduction learning
                 reduce_lin=False,
                 monitor='val_loss',
                 **kwargs):

        ## Custom modification:  Deprecated
        # super(ReduceLROnPlateau, self).__init__()

        ## Custom modification:  Deprecated
        # self.monitor = monitor

        ## Custom modification: Optimizer Error Handling
        if tf.is_tensor(optim_lr) == False:
            raise ValueError('Need optimizer !')
        if factor >= 1.0:
            raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        ## Custom modification: Passing optimizer as arguement
        self.optim_lr = optim_lr

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.mode = mode
        self.monitor_op = None
        self.sign_number = sign_number
        self.monitor = monitor

        ## Custom modification: linearly reducing learning
        self.reduce_lin = reduce_lin
        self.reduce_lr = True

        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if self.mode not in ['auto', 'min', 'max']:
            print('Learning Rate Plateau Reducing mode %s is unknown, '
                  'fallback to auto mode.', self.mode)
            self.mode = 'auto'
        if (self.mode == 'min' or
                ## Custom modification: Deprecated due to focusing on validation loss
                # (self.mode == 'auto' and 'acc' not in self.monitor)):
                (self.mode == 'auto')):
            self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
            self.best = np.Inf
        else:
            self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
            self.best = -np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ## Custom modification: Optimizer
        # logs['lr'] = K.get_value(self.model.optimizer.lr) returns a numpy array
        # and therefore can be modified to
        logs['lr'] = float(self.optim_lr.numpy())

        ## Custom modification: Deprecated due to focusing on validation loss
        current = logs.get(self.monitor)

        # current = float(logs['val_loss'])

        ## Custom modification: Deprecated due to focusing on validation loss
        # if current is None:
        #     print('Reduce LR on plateau conditioned on metric `%s` '
        #                     'which is not available. Available metrics are: %s',
        #                     self.monitor, ','.join(list(logs.keys())))

        # else:

        if self.in_cooldown():
            self.cooldown_counter -= 1
            self.wait = 0

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0

        elif not self.in_cooldown():
            self.wait += 1
            if self.wait >= self.patience:

                ## Custom modification: Optimizer Learning Rate
                # old_lr = float(K.get_value(self.model.optimizer.lr))
                old_lr = float(self.optim_lr.numpy())
                if old_lr > self.min_lr and self.reduce_lr == True:
                    ## Custom modification: Linear learning Rate
                    if self.reduce_lin == True:
                        new_lr = old_lr - self.factor
                        ## Custom modification: Error Handling when learning rate is below zero
                        if new_lr <= 0:
                            print('Learning Rate is below zero: {}, '
                                  'fallback to minimal learning rate: {}. '
                                  'Stop reducing learning rate during training.'.format(new_lr, self.min_lr))
                            self.reduce_lr = False
                    else:
                        new_lr = old_lr * self.factor

                    new_lr = max(new_lr, self.min_lr)

                    ## Custom modification: Optimizer Learning Rate
                    # K.set_value(self.model.optimizer.lr, new_lr)
                    self.optim_lr.assign(new_lr)

                    if self.verbose > 0:
                        print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                              'rate to %s.' % (epoch + 1, float(new_lr)))
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=0, min_delta=0, verbose=0, mode='min', start_from_epoch=0, monitor='val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor = monitor
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.mode = mode
        self.best_epoch = 0
        self.start_from_epoch = start_from_epoch

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def _reset(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_epoch = 0

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

    def on_epoch_end(self, epoch, logs=None):
        global early_stop
        current = logs.get(self.monitor)

        self.wait += 1

        # Monitor if val loss improved and if so save weights and reset wait counter
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.wait = 0

        # End condition and restoring weights to mode;
        if self.wait >= self.patience and epoch > 0:
            self.stopped_epoch = epoch
            early_stop = True
            print(f"\nEpoch {epoch + 1}: EarlyStopping...")

            return True

        return False

    def on_train_begin(self, logs=None):
        self._reset()
