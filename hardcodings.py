import os
from os.path import join as path_join


PROJECT_PATH = "D:\\Igno\\Reikalingi\\KTU Magistras\\Magistrinis\\OoMLeT"
EXPERIMENTS_PATH = path_join(PROJECT_PATH, "experiments")
DATA_FILTERING_PATH = path_join(PROJECT_PATH, "filtered_data")
SEARCHED_SPACE_PATH = path_join(PROJECT_PATH, "searched_space")
EXPERIMENT_CONFIGS_PATH = path_join(PROJECT_PATH, "exp_configs")

RESULTS_CSV_NAME = "results.csv"
TRAINING_CSV_NAME = 'training.csv'

MIN_CORRECT_LOSS = 0.3