import os
from datetime import datetime
from os.path import join as path_join
from typing import Optional

import numpy as np
import pandas as pd
from Common.data_manipulation.pandas_tools import save_dataframe, load_df
from Common.data_manipulation.pickler import load_pickle
from Common.data_manipulation.string_tools import get_files_list_in_dir_with_substrings
from Common.machine_learning_tools.qol_machine_learning_tools import get_spent_time
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

from config_space import parse_config_space, load_config, parse_bounds_dict, format_suggested_config
from data_processing.data_filtering import filter_dataset
from data_processing.data_loaders import load_defined_data
from data_processing.simple_generator import SimpleGenerator
from experiment_runner import run_training
from hardcodings import EXPERIMENTS_PATH, DATA_FILTERING_PATH, SEARCHED_SPACE_PATH, EXPERIMENT_CONFIGS_PATH, \
    RESULTS_CSV_NAME


def main(config_file_name: str, experiment_config_overrides: Optional[dict] = None) -> None:
    experiment_config = load_config(config_file_name)
    override_config(experiment_config, experiment_config_overrides)

    target_metric = "test_acc"
    # experiment_config['max_runs'] = 1
    # experiment_config['epochs'] = 1
    filtering_frequency = experiment_config.get('filtering_frequency', 0)
    batch_size = experiment_config.get("batch_size", 128)

    dataset_solution_name = "{}-{}".format(experiment_config['dataset'], experiment_config['architecture'])
    dataset_solution_path = path_join(EXPERIMENTS_PATH, dataset_solution_name)
    os.makedirs(dataset_solution_path, exist_ok=True)

    homogenous_experiments = get_files_list_in_dir_with_substrings(dataset_solution_path, [experiment_config['name']])
    if len(homogenous_experiments) > 0:
        experiment_config['name'] += "-{}".format(len(homogenous_experiments) + 1)

    experiment_path = path_join(dataset_solution_path, experiment_config['name'])
    os.makedirs(experiment_path, exist_ok=True)
    result_csv_fullname = path_join(experiment_path, RESULTS_CSV_NAME)

    dataset = load_defined_data(experiment_config['dataset'], experiment_config['data_format'], do_preprocess=True)

    dataset['train_indice'] = np.arange(len(dataset['x_train']))
    train_indice = delete_easy_samples(dataset, experiment_config)

    experiment_config['input_shape'] = dataset['input_shape']
    experiment_config['num_classes'] = dataset['num_classes']

    train_generator = SimpleGenerator(dataset['x_train'], dataset['y_train'], dataset['train_indice'], batch_size=batch_size, preprocess=False)
    val_generator = SimpleGenerator(dataset['x_val'], dataset['y_val'], batch_size=batch_size, preprocess=False)
    test_generator = SimpleGenerator(dataset['x_test'], dataset['y_test'], batch_size=batch_size, preprocess=False)

    # Define search space
    search_space = parse_config_space(experiment_config)
    bounds_dict = parse_bounds_dict(search_space)
    hp_names = bounds_dict.keys()

    optimizer = BayesianOptimization(
        f=None,
        pbounds=bounds_dict,
        verbose=2,
    )

    load_searched_space(optimizer, experiment_config, hp_names, target_metric)

    utility = UtilityFunction(kind=experiment_config["utility_f"], kappa=experiment_config["kappa"], xi=experiment_config["xi"])

    run_results = []
    for run_idx in range(experiment_config["max_runs"]):
        suggested_config = optimizer.suggest(utility)

        run_config, hyperparameters = format_suggested_config(suggested_config, experiment_config)
        print("Next point ({:2d}) to probe is: {}".format(run_idx, hyperparameters))

        run_path = path_join(experiment_path, str(run_idx))
        os.makedirs(run_path, exist_ok=True)

        now = datetime.now()
        metrics = run_training(run_config, train_generator, val_generator, test_generator, run_path)
        now, diff = get_spent_time(now, False)
        target = metrics[target_metric]
        print(metrics)
        print("Found the target value to be: {}".format(target))

        optimizer.register(suggested_config, target)

        run_result = hyperparameters.copy()
        run_result.update(metrics)
        run_result['time_spent'] = diff.seconds
        run_result['train_set_size'] = len(train_generator.X_data)
        run_results.append(run_result)

        result_df = pd.DataFrame.from_dict(run_results)
        save_dataframe(result_df, result_csv_fullname)

        if filtering_frequency > 0 and (run_idx+1) % filtering_frequency == 0:
            filtered_x_train, filtered_y_train, filtered_train_indice = filter_dataset(
                dataset['x_train'], dataset['y_train'], dataset['train_indice'], experiment_path, experiment_config, run_idx, target_metric
            )
            train_generator = SimpleGenerator(filtered_x_train, filtered_y_train, filtered_train_indice, batch_size=batch_size, preprocess=False)

        print("Run end")
        print()

    print("Optimization end")

    # result_df = pd.DataFrame.from_dict(run_results)
    # result_csv_fullname = path_join(experiment_path, "results.csv")
    # save_dataframe(result_df, result_csv_fullname)


def override_config(experiment_config, experiment_config_overrides):
    if experiment_config_overrides:
        print("Received experiment_config_overrides:")
        print(experiment_config_overrides)
        experiment_config.update(experiment_config_overrides)

        for key, value in experiment_config_overrides.items():
            key_abbreviation = "".join([key_part[0] for key_part in key.split('_')])
            experiment_config['name'] += "-" + key_abbreviation + str(value).replace('.','')


def load_searched_space(optimizer, experiment_config, hp_names, target_metric):
    # Load searched space
    searched_space_csv_name = experiment_config.get('load_ss', None)
    if not searched_space_csv_name:
        return None

    searched_space_csv_fullname = path_join(SEARCHED_SPACE_PATH, searched_space_csv_name)
    print("Loading searched space")
    searched_space_df = load_df(searched_space_csv_fullname)

    for index, pandas_row in searched_space_df.iterrows():
        searched_config = pandas_row[hp_names]
        target_metric_value = pandas_row[target_metric]

        optimizer.register(searched_config, target_metric_value)
        print("Registered {} with value {:.3f}".format(searched_config, target_metric_value))


def delete_easy_samples(dataset, experiment_config):
    if experiment_config.get('sample_filter', None):
        sample_filter_fullname = path_join(DATA_FILTERING_PATH, experiment_config['sample_filter'])
        sample_filter = load_pickle(sample_filter_fullname)

        dataset['train_indice'] = np.delete(dataset['train_indice'], sample_filter, axis=0)
        dataset['x_train'] = np.delete(dataset['x_train'], sample_filter, axis=0)
        dataset['y_train'] = np.delete(dataset['y_train'], sample_filter, axis=0)

        print("Deleted {} easy samples".format(len(sample_filter)))


if __name__ == '__main__':
    # config_file_fullnames = [
    #     "exp_configs/MNIST-LeNet2-config.json",
    #     "exp_configs/MNIST-LeNet2-config-2.json",
    #     "exp_configs/MNIST-LeNet2-config-3.json",
    # ]
    config_file_fullname = path_join(EXPERIMENT_CONFIGS_PATH, "CIFAR-10", "config-1.json")

    dataset_filtering_frequency_and_quantile = [
        # (3, 0.5), (5, 0.5),
        (5, 0.7), (10, 0.5), (10, 0.7),
    ]
    kappas = [1.0, 2.5, 4.0]

    for i in range(5):
        for filtering_frequency, filtering_quantile in dataset_filtering_frequency_and_quantile:
            for kappa in kappas:
                print("Running", config_file_fullname)

                experiment_config_overrides = {
                    "kappa": kappa,
                    "filtering_frequency": filtering_frequency,
                    "filtering_quantile": filtering_quantile,
                }

                try:
                    main(config_file_fullname, experiment_config_overrides)
                except Exception as e:
                    print("--------------------------- EXCEPTION ---------------------------")
                    print(e)
                    print("--------------------------- EXCEPTION ---------------------------")

                finally:
                    continue


    # config_file_fullnames = [path_join(EXPERIMENT_CONFIGS_PATH, "CIFAR-10", config_name)
    #                          for config_name in
    #                          sorted(os.listdir(path_join(EXPERIMENT_CONFIGS_PATH, "CIFAR-10")))]
    #
    # for i in range(5):
    #     for config_file_fullname in config_file_fullnames:
    #         print("Running", config_file_fullname)
    #         main(config_file_fullname)

            # try:
            #     main(config_file_fullname)
            # except Exception as e:
            #     print("--------------------------- EXCEPTION ---------------------------")
            #     print(e)
            #     print("--------------------------- EXCEPTION ---------------------------")
            #
            # finally:
            #     continue

    # while True:
    #     # main(config_file_fullnames[0])
    #
    #     for config_file_fullname in config_file_fullnames:
    #         print("Running", config_file_fullname)
    #         main(config_file_fullname)
            # try:
            #     main(config_file_fullname)
            # except Exception as e:
            #     print("--------------------------- EXCEPTION ---------------------------")
            #     print(e)
            #     print("--------------------------- EXCEPTION ---------------------------")
            #
            # finally:
            #     continue
