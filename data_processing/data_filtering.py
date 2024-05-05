import os
from os.path import join as path_join

import numpy as np
import pandas as pd
from Common.data_manipulation.dictionary_tools import create_or_append_to_dict_of_lists
from Common.data_manipulation.pandas_tools import load_df
from Common.data_manipulation.pickler import write_pickle
from Common.visualizations.figure_plotting import plot_histogram

from hardcodings import RESULTS_CSV_NAME, TRAINING_CSV_NAME, MIN_CORRECT_LOSS


def filter_dataset(x_train, y_train, train_indice, experiment_path, experiment_config, run_index=0, result_metric='test_acc'):
    filtering_quantile = experiment_config.get('filtering_quantile', 0.5)
    filtering_method = experiment_config.get('filtering_method', "example forgetting")

    csv_fullname = path_join(experiment_path, RESULTS_CSV_NAME)
    res_df = load_df(csv_fullname)

    median_acc = res_df[result_metric].quantile(filtering_quantile)
    good_df = res_df.loc[res_df[result_metric] >= median_acc]
    good_indice = list(good_df.index)

    full_df = load_experiment_runs_results(experiment_path, good_indice)
    runs_sample_losses = runs_results_to_dict_of_losses(full_df)

    if filtering_method == "example forgetting":
        samples_runs_forget_counts, runs_samples_forget_counts = count_forgetting_events(runs_sample_losses)

        runs_sample_forgetting_df = pd.DataFrame.from_dict(samples_runs_forget_counts, orient='index', columns=runs_samples_forget_counts.keys())

        max_forgets = runs_sample_forgetting_df.max(axis=1)

        max_forgets_plot_name = "max_forgets-{}.jpg".format(run_index)
        max_forgets_plot_fullname = path_join(experiment_path, max_forgets_plot_name)
        plot_histogram(max_forgets, add_cumulative=True, save_fullname=max_forgets_plot_fullname, only_save=True)

        unforgettable_samples = list(max_forgets[max_forgets == 0].index)
        forgettable_samples = list(max_forgets[max_forgets > 0].index)

        pickle_fullname = path_join(experiment_path, 'unforgettable_samples-{}.pckl'.format(run_index))
        write_pickle(pickle_fullname, unforgettable_samples)

        filtered_x_train = x_train[forgettable_samples]
        filtered_y_train = y_train[forgettable_samples]
        filtered_train_indice = train_indice[forgettable_samples]

        print("Epoch {} | Found {} unforgettable samples | Remaining train set size: {}".format(
            run_index, len(unforgettable_samples), len(filtered_train_indice)))

    else:
        raise NotImplementedError("Dataset filtering method {} is not implemented".format(filtering_method))

    return filtered_x_train, filtered_y_train, filtered_train_indice


def df_to_index_loss_dict(df):
    index_to_loss_dict = dict(tuple(df.groupby('index')['loss']))
    index_to_loss_dict = {key: np.array(list(value)) for key, value in index_to_loss_dict.items()}

    return index_to_loss_dict


def load_experiment_runs_results(experiment_path, good_run_incide):
    dfs = []

    for run_index in good_run_incide:
        train_csv_fullname = path_join(experiment_path, str(run_index), TRAINING_CSV_NAME)
        if not os.path.isfile(train_csv_fullname):
            continue

        df = load_df(train_csv_fullname)
        df['run'] = int(run_index)

        dfs.append(df)

    full_df = pd.concat(dfs)
    # full_df.sort_values(['run', 'epoch', 'index'], inplace=True)

    return full_df


def runs_results_to_dict_of_losses(runs_results_df):
    runs_sample_losses = {}

    for run_idx in pd.unique(runs_results_df['run']):
        run_df = runs_results_df.loc[runs_results_df['run'] == run_idx]

        # print(run_idx, len(run_df))

        index_to_loss_dict = df_to_index_loss_dict(run_df)

        runs_sample_losses[run_idx] = index_to_loss_dict

    return runs_sample_losses


def count_forgetting_events(runs_sample_losses, min_correct_loss=MIN_CORRECT_LOSS):
    samples_runs_forget_counts = {}
    runs_samples_forget_counts = {}

    for run_index, run_losses in runs_sample_losses.items():
        samples_forget_counts = {}

        for sample_index, sample_losses in run_losses.items():
            forgetting_events = np.bitwise_and(sample_losses[:-1] < sample_losses[1:],
                                               sample_losses[1:] > min_correct_loss)
            forgetting_events_count = np.sum(forgetting_events)

            samples_forget_counts[sample_index] = forgetting_events_count
            create_or_append_to_dict_of_lists(samples_runs_forget_counts, sample_index, forgetting_events_count)

        runs_samples_forget_counts[run_index] = samples_forget_counts

    return samples_runs_forget_counts, runs_samples_forget_counts