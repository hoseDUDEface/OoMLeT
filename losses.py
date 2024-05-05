import tensorflow as tf


class SampleWiseCatCrossentropy(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name='CustomLoss'):
        super(SampleWiseCatCrossentropy, self).__init__(reduction=reduction, name=name)

        # self.batch_index = tf.Variable(0.)
        # self.loss_metric_list = []

    def call(self, y_true, y_pred):
        sample_losses = tf.losses.sparse_categorical_crossentropy(y_true, y_pred)

        return sample_losses



    # def get_loss_metric_list(self):
    #     return self.loss_metric_list
    #
    # def get_metrics_states_dict(self):
    #     metrics_states_dict = {metric.name: metric.result() for metric in self.loss_metric_list}
    #
    #     return metrics_states_dict
    #
    # def reset_metrics(self):
    #     self.reset_log()
    #
    #     for metric in self.loss_metric_list:
    #         metric.reset_state()
    #
    # def reset_log(self):
    #     self.batch_index.assign(0.)