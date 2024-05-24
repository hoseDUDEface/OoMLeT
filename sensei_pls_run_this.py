from optimisation_loop import main

if __name__ == '__main__':

    experiment_tuples = [
        ("exp_configs/CIFAR-10/config-1st_L.json",     [(1, 10), (2.5, 10), (4, 10)]),
        ("exp_configs/CIFAR-10/config-EF.json",        [(1, 3), (1, 5), (1, 2), (1, 10),
                                                        (2.5, 10),
                                                        (4, 5), (4, 10)]),


        # ("exp_configs/MNIST/LeNet1-config-1st_L.json", [(2.5, 5), (25, 2), (4, 5), (4, 2)]),
        # ("exp_configs/MNIST/LeNet1-config-EF.json",    [(1, 3), (1, 5), (1, 2), (1, 10),
        #                                                 (2.5, 3), (2.5, 2), (2.5, 10),
        #                                                 (4, 3), (4, 5), (4, 3), (4, 10)]),
        #
        # ("exp_configs/MNIST/LeNet2-config-1st_L.json", [(1, 5), (1, 2), (1, 10),
        #                                                 (4, 3), (4, 5), (4, 2)]),
        # ("exp_configs/MNIST/LeNet2-config-EF.json",    [(1, 2), (1, 2),
        #                                                 (2.5, 2), (2.5, 10),
        #                                                 (4, 10)]),
    ]

    for i in range(5):
        for config_file_fullname, param_tuples in experiment_tuples:
            for kappa, top_k in param_tuples:
                print("Running", config_file_fullname)

                experiment_config_overrides = {
                    "kappa": kappa,
                    "filtering_top_k": top_k,
                }

                try:
                    main(config_file_fullname, experiment_config_overrides)
                except Exception as e:
                    print("--------------------------- EXCEPTION ---------------------------")
                    print(e)
                    print("--------------------------- EXCEPTION ---------------------------")

                finally:
                    continue

