import optuna

class SearchSpace:
    """
    base class for search spaces
    """

    defaults = None # override this

    @staticmethod
    def get_params():
        raise NotImplementedError()


# class pnet1_v1(SearchSpace):

#     defaults = {
#         # main
#         "output-flow": "seg",
#         "loss": "gridmse",
#         # data
#         "subdivide": 5,
#         "noise-sigma": 0.1,
#         # training
#         "optimizer": "adam",
#         "batchsize-exp": 4,
#         "lr-exp": -1.5,
#         # model arch
#         "npoints-exp": 3,
#         "sm-exp": 0,
#         "seg-dropout": 0.2,
#         # loss
#         "gaussian-sigma": 2.5,
#         "ortho-weight-exp": -2,
#     }

#     @staticmethod
#     def get_params(args, trial):
#         flags = []
#         params = {
#             # main
#             "output-flow": trial.suggest_categorical("output-flow", ["dense", "seg"]),
#             "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
#             # data
#             "subdivide": trial.suggest_int("subdivide", 1, 5),
#             "noise-sigma": trial.suggest_float("noise-sigma", 0.0, 3.0, step=0.1), # in meters
#             # training
#             "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
#             "batchsize": 2 ** trial.suggest_int("batchsize-exp", 2, 7), # 4 to 128
#             "lr": 10 ** trial.suggest_float("lr-exp", -5, -1, step=0.5), # 1e-1 to 1e-5
#             # model arch
#             "npoints": 100 * 2 ** trial.suggest_int("npoints-exp", 1, 4), # 200 to 3200
#             "size-multiplier": 2 ** trial.suggest_int("sm-exp", -1, 3), # 0.5 to 8
#             # loss
#             "gaussian-sigma": trial.suggest_float("gaussian-sigma", 1, 8, step=0.5) # in meters
#         }
#         if params["loss"] == "mmd":
#             params["mmd-kernel"] = "gaussian"
#         elif params["loss"] == "gridmse":
#             params["grid-agg"] = trial.suggest_categorical("grid-agg", ["sum", "max"])
        
#         if params["output-flow"] == "dense":
#             params["out-npoints"] = 2 ** trial.suggest_int("out-npoints-exp", 7, 10) # 128 to 2048
#             params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.7, step=0.05)
#         elif params["output-flow"] == "seg":
#             params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.7, step=0.05)

#         # not using TNets

#         return params, flags


# class pnet2_v1(SearchSpace):

#     # defaults are fed in as trial params to the get_params function on the 1st trial
#     defaults = {
#         # main
#         "pnet2": True,
#         "output-flow": "seg",
#         "loss": "gridmse",
#         # data
#         "subdivide": 5,
#         "noise-sigma": 0.1,
#         # training
#         "optimizer": "adam",
#         "batchsize-exp": 4,
#         "lr-exp": -1.5,
#         # model arch
#         "npoints-exp": 3,
#         "size-multiplier": 1.0,
#         "seg-dropout": 0.2,
#         "batchnorm": False,
#         # loss
#         "gaussian-sigma": 2.5,
#     }


#     @staticmethod
#     def get_params(args, trial):
#         flags = [
#             "pnet2",
#         ]
#         params = {
#             # main
#             "output-flow": trial.suggest_categorical("output-flow", ["dense", "seg"]),
#             "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
#             # data
#             "subdivide": trial.suggest_int("subdivide", 1, 6),
#             "noise-sigma": trial.suggest_float("noise-sigma", 0.0, 3.0, step=0.1), # in meters
#             # training
#             "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
#             "batchsize": 2 ** trial.suggest_int("batchsize-exp", 2, 7), # 4 to 128
#             "lr": 10 ** trial.suggest_float("lr-exp", -5, -1, step=0.5), # 1e-1 to 1e-5
#             "reducelr-patience": 3,
#             # model arch
#             "npoints": 100 * 2 ** trial.suggest_int("npoints-exp", 1, 4), # 200 to 3200
#             "size-multiplier": trial.suggest_float("size-multiplier", 0.25, 2, step=0.25),
#             # loss
#             "gaussian-sigma": trial.suggest_float("gaussian-sigma", 1, 8, step=0.5) # in meters
#         }
#         if params["loss"] == "mmd":
#             params["mmd-kernel"] = "gaussian"
#         elif params["loss"] == "gridmse":
#             params["grid-agg"] = trial.suggest_categorical("grid-agg", ["sum", "max"])
        
#         if params["output-flow"] == "dense":
#             params["out-npoints"] = 2 ** trial.suggest_int("out-npoints-exp", 7, 10) # 128 to 2048
#             params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.7, step=0.05)
#         elif params["output-flow"] == "seg":
#             params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.7, step=0.05)

#         if trial.suggest_categorical("batchnorm", [True, False]):
#             flags.append("batchnorm")

#         return params, flags


class full(SearchSpace):

    # defaults are fed in as trial params to the get_params function on the 1st trial
    defaults = {
        "pnet2": True,
        # data
        "subdivide": 5,
        "npoints-exp": 3,
        "noise-sigma": 0.1,
        "handle-small": "fill",
        # training
        "optimizer": "adam",
        "batchsize-exp": 4,
        "lr-exp": -1.5,
        # model arch
        "output-flow": "seg",
        "sm-exp": 0,
        "conf-act": "sigmoid",
        "seg-dropout": 0.2,
        "batchnorm": False,
        # loss
        "loss": "gridmse",
        "gridmse-agg": "max",
        "gaussian-sigma": 2.5,
    }


    @staticmethod
    def get_params(args, trial):
        flags = []
        params = {
            # data
            "subdivide": trial.suggest_int("subdivide", 1, 7),
            "npoints": 100 * 2 ** trial.suggest_int("npoints-exp", 1, 4), # 200 to 3200
            "noise-sigma": trial.suggest_float("noise-sigma", 0.0, 2.0, step=0.1), # in meters
            "handle-small": trial.suggest_categorical("handle-small", ["drop", "fill", "repeat"]),
            # training
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
            "batchsize": 2 ** trial.suggest_int("batchsize-exp", 3, 7), # 8 to 128
            "lr": 10 ** trial.suggest_float("lr-exp", -5, -1, step=0.5), # 1e-1 to 1e-5
            "reducelr-patience": 3,
            # model arch
            "output-flow": trial.suggest_categorical("output-flow", ["dense", "seg"]),
            "size-multiplier": 2 ** trial.suggest_int("sm-exp", -1, 2), # 0.5 to 4
            "conf-act": trial.suggest_categorical("conf-act", ["relu", "sigmoid"]),
            # loss
            "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
            "gaussian-sigma": trial.suggest_float("gaussian-sigma", 0.5, 6, step=0.5) # in meters
        }
        # losses
        if params["loss"] == "mmd":
            params["mmd-kernel"] = "gaussian"
        elif params["loss"] == "gridmse":
            params["gridmse-agg"] = trial.suggest_categorical("gridmse-agg", ["sum", "max"])
        
        # output flow
        if params["output-flow"] == "dense":
            params["out-npoints"] = 2 ** trial.suggest_int("out-npoints-exp", 7, 10) # 128 to 2048
            params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.7, step=0.05)
        elif params["output-flow"] == "seg":
            params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.7, step=0.05)

        # pnet 1 vs 2
        if trial.suggest_categorical("pnet2", [False, True]):
            flags.append("pnet2")
            if trial.suggest_categorical("batchnorm", [True, False]):
                flags.append("batchnorm")
        

        return params, flags


