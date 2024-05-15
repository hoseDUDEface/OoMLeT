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
from data_processing.data_filterer import DataFilterer
from data_processing.data_filtering import filter_dataset
from data_processing.data_loaders import load_defined_data
from data_processing.simple_generator import SimpleGenerator
from experiment_runner import run_training
from hardcodings import EXPERIMENTS_PATH, DATA_FILTERING_PATH, SEARCHED_SPACE_PATH, EXPERIMENT_CONFIGS_PATH, \
    RESULTS_CSV_NAME


def main(config_file_name: str) -> None:
    experiment_config = load_config(config_file_name)

    target_metric = "test_acc"
    batch_size = experiment_config.get("batch_size", 128)

    experiment_path = os.path.split(config_file_name)[0]

    # Dataset processing
    dataset = load_defined_data(experiment_config['dataset'], experiment_config['data_format'], do_preprocess=True)

    dataset['train_indice'] = np.arange(len(dataset['x_train']))
    # dataset['train_indice'] = delete_easy_samples(dataset, experiment_config)

    experiment_config['input_shape'] = dataset['input_shape']
    experiment_config['num_classes'] = dataset['num_classes']

    train_generator = SimpleGenerator(dataset['x_train'], dataset['y_train'], dataset['train_indice'], batch_size=batch_size, preprocess=False)
    val_generator = SimpleGenerator(dataset['x_val'], dataset['y_val'], batch_size=batch_size, preprocess=False)
    test_generator = SimpleGenerator(dataset['x_test'], dataset['y_test'], batch_size=batch_size, preprocess=False)

    now = datetime.now()
    metrics = run_training(experiment_config, train_generator, val_generator, test_generator, experiment_path)
    now, diff = get_spent_time(now, False)
    print("Experiment took", diff.seconds)
    target = metrics[target_metric]
    print(metrics)
    print("Found the target value to be: {}".format(target))

    print("Run end")


if __name__ == '__main__':
    # config_file_fullnames = [
    #     "exp_configs/MNIST/LeNet1-config-EF.json",
    #     "exp_configs/MNIST/LeNet2-config-EF.json",
    # ]
    # config_file_fullname = path_join(EXPERIMENT_CONFIGS_PATH, "CIFAR-10", "config-2.json")
    # config_file_fullname = path_join(EXPERIMENT_CONFIGS_PATH, "MNIST", "MNIST-LeNet1-config-algo.json")

    config_file_name = "MNIST-LeNet1\\topk\\example_forgetting-ucb-k10-ftk5\\47\\model_config.json"
    config_file_fullname = path_join(EXPERIMENTS_PATH, config_file_name)
    main(config_file_fullname)