import json
from typing import Union

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np


def parse_config_space(ss_config: Union[dict, str]) -> CS.ConfigurationSpace:
    """
    hp_types: "uniform-f", "uniform-i"
    hp_type, hp_min, hp_max, hp_default = '', 0, 0, None
    """
    if ss_config is str:
        with open(ss_config, 'r') as f:
            ss_config = json.load(f)

    cs = CS.ConfigurationSpace()

    for hp_name, hp_value in ss_config.items():
        hp_type, hp_min, hp_max, hp_log = '', 0, 0, False

        if type(hp_value) is not list:
            continue

        if hp_value[0] == 'categorical':
            cs_hp = CSH.CategoricalHyperparameter(hp_name, hp_value[1])

        else:
            if len(hp_value) == 3:
                hp_type, hp_min, hp_max = hp_value

            elif len(hp_value) == 4:
                hp_type, hp_min, hp_max, hp_log = hp_value

            # if 'depth' in hp_name:
            #     # Means this is layer depth, which should be divisible by 8
            #     hp_min = max(1, int(hp_min//8))
            #     hp_max = int(np.round(hp_max//8))

            if hp_type.lower() == "uniform-f":
                cs_hp = CSH.UniformFloatHyperparameter(hp_name, hp_min, hp_max, log=hp_log)

            elif hp_type.lower() == "uniform-i":
                cs_hp = CSH.UniformIntegerHyperparameter(hp_name, hp_min, hp_max, log=hp_log)

            else:
                raise NotImplemented("{} hp type {} not implemented".format(hp_name, hp_type))

        cs.add_hyperparameter(cs_hp)

    print("Search space to explore:")
    print(cs)

    return cs


def load_config(config_file_name):
    with open(config_file_name, 'r') as f:
        run_config = json.load(f)
    return run_config


def parse_bounds_dict(config_space: CS.ConfigurationSpace):
    bounds_dict = {}

    for hp in list(config_space.values()):
        if isinstance(hp, CSH.CategoricalHyperparameter):
            bounds_dict[hp.name] = (0, hp.num_choices-1)

        elif hp.log:
            bounds_dict[hp.name] = (np.log10(hp.lower), np.log10(hp.upper))

        elif 'depth' in hp.name:
            # Means this is layer depth, which should be divisible by 8
            hp_min = max(1, int(hp.lower//8))
            hp_max = int(np.round(hp.upper//8))

            bounds_dict[hp.name] = (hp_min, hp_max)

        else:
            bounds_dict[hp.name] = (hp.lower, hp.upper)

    return bounds_dict


def format_suggested_config(suggested_config, experiment_config):
    new_config = {}
    hyperparameters = {}

    for name, value in experiment_config.items():
        if name in suggested_config:
            suggested_value = suggested_config[name]

            if type(value) is list:
                if value[0] == 'categorical':
                    category_index = int(np.round(suggested_value))
                    hp_value = value[1][category_index]
                elif len(value) == 4 and value[3]:
                    # Means this hp is log
                    hp_value = 10 ** suggested_value
                    # print("Scaled {} from {:.4f} to {:.4f}".format(name, suggested_value, hp_value))
                elif 'depth' in name:
                    # Means this is layer depth, which should be divisible by 8
                    hp_value = int(np.round(hp_value) * 8)
                elif value[0][-1] == 'i':
                    # Means this hp is integer
                    hp_value = int(np.round(suggested_value))
                else:
                    hp_value = suggested_value

                new_config[name] = hp_value
                hyperparameters[name] = hp_value

            else:
                new_config[name] = suggested_value

        else:
            new_config[name] = value

    return new_config, hyperparameters



