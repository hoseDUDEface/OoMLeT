from optimisation_loop import main

if __name__ == '__main__':
    config_file_fullnames = [
        "exp_configs/CIFAR-10/config-1st_L.json",
    ]

    top_ks = [3, 5, 2, 10]
    kappas = [
        2.5,
        4.0,
        1.0,
    ]

    for i in range(5):
        for top_k in top_ks:
            for config_file_fullname in config_file_fullnames:
                for kappa in kappas:
                    print("Running", config_file_fullname)

                    experiment_config_overrides = {
                        "kappa": kappa,
                        "filtering_top_k": top_k,
                    }
                    main(config_file_fullname, experiment_config_overrides)
                    #
                    # try:
                    #     main(config_file_fullname, experiment_config_overrides)
                    # except Exception as e:
                    #     print("--------------------------- EXCEPTION ---------------------------")
                    #     print(e)
                    #     print("--------------------------- EXCEPTION ---------------------------")
                    #
                    # finally:
                    #     continue