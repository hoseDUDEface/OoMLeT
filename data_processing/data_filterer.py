import os
from os.path import join as path_join

import numpy as np
import pandas as pd
from Common.data_manipulation.dictionary_tools import create_or_append_to_dict_of_lists
from Common.data_manipulation.pandas_tools import load_df
from Common.data_manipulation.pickler import write_pickle
from Common.visualizations.figure_plotting import plot_histogram
from sklearn.metrics import log_loss

from hardcodings import RESULTS_CSV_NAME, TRAINING_CSV_NAME


class DataFilterer:
    def __init__(self, dataset, experiment_config, experiment_path, target_metric, baseline_acc=0, min_correct_loss=None):
        self.dataset = dataset
        self.experiment_config = experiment_config
        self.experiment_path = experiment_path
        self.target_metric = target_metric
        self.baseline_acc = baseline_acc
        self.min_correct_loss = min_correct_loss

        self.x_train = dataset['x_train']
        self.y_train = dataset['y_train']
        self.train_indice = dataset['train_indice']

        # self.filtering_frequency = experiment_config.get('filtering_frequency', 0)
        self.filtering_quantile = experiment_config.get('filtering_quantile', 0)
        self.filtering_top_k = experiment_config.get('filtering_top_k', 0)
        self.filtering_method = experiment_config.get('filtering_method', None)
        self.filtering_max_forgets = experiment_config.get('filtering_max_forgets', 0)
        self.filtering_aggregation = experiment_config.get('filtering_aggregation', 'max')
        # Do not filter if fields are not defined
        self.do_filter = (self.filtering_quantile or self.filtering_top_k) and self.filtering_method
        if not self.do_filter: print("Filtering values not defined, will not filter")

        self.result_csv_fullname = path_join(experiment_path, RESULTS_CSV_NAME)
        self.result_df = None
        self.last_run_idx = -1
        self.last_run_relevant_indice = []
        self.forgettable_samples = None
        self.minimum_runs_to_filter = max(2, self.filtering_top_k)

        if self.min_correct_loss is None:
            self.min_correct_loss = self.calculate_minimum_correct_loss()

    def update_result_df(self):
        self.result_df = load_df(self.result_csv_fullname)

    def do_filter_condition(self, run_index):
        # Wait at least self.minimum_runs_to_filter runs before filtering
        if self.do_filter and run_index >= self.minimum_runs_to_filter:
            relevant_indice = self.get_relevant_run_indice()
            good_result_df = self.result_df.loc[self.result_df[self.target_metric] > self.baseline_acc]

            if len(good_result_df) <= self.minimum_runs_to_filter:
                print("Not enough runs that beat baseline")
                return False

            elif len(relevant_indice) < 2 or (0 < self.filtering_top_k and self.filtering_top_k > len(relevant_indice)):
                print("Not enough relevant_indice")
                return False

            elif relevant_indice == self.last_run_relevant_indice:
                print("Would filter based on same runs as last time, so skipping filtering processing")
                return False

            else:
                return True

        return False

    def get_filtered_dataset(self, run_index):
        relevant_indice = self.get_relevant_run_indice()

        trainings_df = self.load_training_csvs(relevant_indice)

        if self.filtering_method == "example forgetting":
            runs_sample_losses = self.trainings_df_to_dict_of_losses(trainings_df)
            forgettable_samples, unforgettable_samples = self.example_forgetting_filter(runs_sample_losses, run_index)

        elif self.filtering_method == "1st look hardness":
            runs_first_losses = self.trainings_df_to_dict_of_losses(trainings_df, True)

            forgettable_samples, unforgettable_samples = self.first_look_hardness_filter(runs_first_losses)

        else:
            raise NotImplementedError("Dataset filtering method {} is not implemented".format(self.filtering_method))

        pickle_fullname = path_join(self.experiment_path, 'unforgettable_samples-{}.pckl'.format(run_index))
        write_pickle(pickle_fullname, unforgettable_samples)

        updated_forgettable_samples = set(forgettable_samples).union(set(self.forgettable_samples)) if self.forgettable_samples else set(forgettable_samples)
        # updated_forgettable_samples = forgettable_samples
        self.forgettable_samples = list(updated_forgettable_samples)
        self.last_run_idx = run_index

        print("Epoch {} | {} found {} unforgettable samples | Remaining train set size: {}".format(
            run_index, self.filtering_method, len(unforgettable_samples), len(self.forgettable_samples)
        ))

        filtered_x_train = self.x_train[self.forgettable_samples]
        filtered_y_train = self.y_train[self.forgettable_samples]
        filtered_train_indice = self.train_indice[self.forgettable_samples]

        return filtered_x_train, filtered_y_train, filtered_train_indice

    def first_look_hardness_filter(self, runs_first_losses):
        samples_runs_first_loss = self.get_samples_runs_first_losses(runs_first_losses)

        if self.filtering_aggregation == 'max':
            aggregated_first_losses = np.array([[key, np.max(values)] for key, values in samples_runs_first_loss.items()])
        elif self.filtering_aggregation == 'min':
            aggregated_first_losses = np.array([[key, np.min(values)] for key, values in samples_runs_first_loss.items()])
        else:
            raise NotImplementedError("filtering_aggregation method {} Not Implemented".format(self.filtering_aggregation))

        unforgettable_samples = aggregated_first_losses[aggregated_first_losses[:, 1] < self.min_correct_loss, 0].astype(int)
        forgettable_samples = aggregated_first_losses[aggregated_first_losses[:, 1] >= self.min_correct_loss, 0].astype(int)

        return forgettable_samples, unforgettable_samples

    @staticmethod
    def get_samples_runs_first_losses(runs_first_losses):
        samples_runs_first_loss = {}

        for run_index, run_losses in runs_first_losses.items():
            for sample_index, sample_loss in run_losses.items():
                create_or_append_to_dict_of_lists(samples_runs_first_loss, sample_index, sample_loss)

        samples_runs_first_loss = {key: np.squeeze(values) for key, values in samples_runs_first_loss.items()}

        return samples_runs_first_loss

    def get_relevant_run_indice(self):
        good_result_df = self.result_df.loc[self.result_df[self.target_metric] > self.baseline_acc]

        if self.filtering_quantile:
            q_acc = good_result_df[self.target_metric].quantile(self.filtering_quantile)
            relevant_df = good_result_df.loc[good_result_df[self.target_metric] >= q_acc]
            relevant_indice = list(relevant_df.index)

            print("Filtered by {:.2f} quantile. Found relevant indice".format(self.filtering_quantile))

        elif self.filtering_top_k:
            top_df = good_result_df.sort_values(self.target_metric, ascending=False).head(self.filtering_top_k)
            relevant_indice = top_df[self.target_metric].index.to_list()

        else:
            relevant_indice = []

        return relevant_indice

    def load_training_csvs(self, relevant_run_indice):
        dfs = []

        for run_index in relevant_run_indice:
            train_csv_fullname = path_join(self.experiment_path, str(run_index), TRAINING_CSV_NAME)
            if not os.path.isfile(train_csv_fullname):
                continue

            df = load_df(train_csv_fullname)
            df['run'] = int(run_index)

            dfs.append(df)

        full_df = pd.concat(dfs)
        # full_df.sort_values(['run', 'epoch', 'index'], inplace=True)

        return full_df

    def example_forgetting_filter(self, runs_sample_losses, run_index):
        samples_runs_forget_counts, runs_samples_forget_counts = self.count_forgetting_events(runs_sample_losses)
        runs_sample_forgetting_df = pd.DataFrame.from_dict(samples_runs_forget_counts, orient='index',
                                                           columns=runs_samples_forget_counts.keys())

        if self.filtering_aggregation == 'max':
            aggregated_forget_counts = runs_sample_forgetting_df.max(axis=1)
        elif self.filtering_aggregation == 'min':
            aggregated_forget_counts = runs_sample_forgetting_df.min(axis=1)
        else:
            raise NotImplementedError("filtering_aggregation method {} Not Implemented".format(self.filtering_aggregation))

        forgets_plot_name = "{}_forgets-{}.jpg".format(self.filtering_aggregation, run_index)
        forgets_plot_fullname = path_join(self.experiment_path, forgets_plot_name)
        plot_histogram(aggregated_forget_counts, add_cumulative=True, save_fullname=forgets_plot_fullname, only_save=True)

        unforgettable_samples = list(aggregated_forget_counts[aggregated_forget_counts < self.filtering_max_forgets].index)
        forgettable_samples = list(aggregated_forget_counts[aggregated_forget_counts >= self.filtering_max_forgets].index)

        return forgettable_samples, unforgettable_samples

    def trainings_df_to_dict_of_losses(self, trainings_df, only_first_epoch=False):
        runs_sample_losses = {}

        for run_idx in pd.unique(trainings_df['run']):
            df_mask = (trainings_df['run'] == run_idx) & (trainings_df['epoch'] == 0) if only_first_epoch else trainings_df['run'] == run_idx
            run_df = trainings_df.loc[df_mask]
            # run_df = trainings_df.loc[trainings_df['run'] == run_idx]

            if only_first_epoch:
                run_df = run_df.drop_duplicates('index')

            index_to_loss_dict = self.df_to_index_loss_dict(run_df)

            runs_sample_losses[run_idx] = index_to_loss_dict

        return runs_sample_losses

    def count_forgetting_events(self, runs_sample_losses):
        samples_runs_forget_counts = {}
        runs_samples_forget_counts = {}

        for run_index, run_losses in runs_sample_losses.items():
            samples_forget_counts = {}

            for sample_index, sample_losses in run_losses.items():
                forgetting_events = np.bitwise_and(sample_losses[:-1] < sample_losses[1:],
                                                   sample_losses[1:] > self.min_correct_loss)
                forgetting_events_count = np.sum(forgetting_events)

                samples_forget_counts[sample_index] = forgetting_events_count
                create_or_append_to_dict_of_lists(samples_runs_forget_counts, sample_index, forgetting_events_count)

            runs_samples_forget_counts[run_index] = samples_forget_counts

        return samples_runs_forget_counts, runs_samples_forget_counts

    def calculate_minimum_correct_loss(self):
        sample_gt = np.zeros(self.dataset['num_classes'])
        sample_gt[0] = 1

        soft_preds = np.array([1 / (self.dataset['num_classes'] * 1.001)] * self.dataset['num_classes'])
        soft_preds[0] += 1 - np.sum(soft_preds)

        min_correct_loss = log_loss(sample_gt, soft_preds)

        return min_correct_loss

    @staticmethod
    def df_to_index_loss_dict(df):
        index_to_loss_dict = dict(tuple(df.groupby('index')['loss']))
        index_to_loss_dict = {key: np.array(list(value)) for key, value in index_to_loss_dict.items()}

        return index_to_loss_dict
