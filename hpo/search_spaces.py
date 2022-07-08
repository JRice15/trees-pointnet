import optuna

class SearchSpace:
    """
    base class for search spaces
    """

    defaults = None # override this

    @staticmethod
    def get_params():
        raise NotImplementedError()


class pnet1_v1(SearchSpace):

    defaults = {
        # main
        "output-mode": "seg",
        "loss": "gridmse",
        # data
        "subdivide": 5,
        "noise-sigma": 0.1,
        # training
        "optimizer": "adam",
        "batchsize-exp": 4,
        "lr-exp": -1.5,
        # model arch
        "npoints-exp": 3,
        "sm-exp": 0,
        "seg-dropout": 0.2,
        # loss
        "gaussian-sigma": 2.5,
        "ortho-weight-exp": -2,
    }

    @staticmethod
    def get_params(args, trial):
        flags = []
        params = {
            # main
            "output-mode": trial.suggest_categorical("output-mode", ["dense", "seg"]),
            "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
            # data
            "subdivide": trial.suggest_int("subdivide", 1, 5),
            "noise-sigma": trial.suggest_float("noise-sigma", 0.0, 3.0, step=0.1), # in meters
            # training
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
            "batchsize": 2 ** trial.suggest_int("batchsize-exp", 2, 7), # 4 to 128
            "lr": 10 ** trial.suggest_float("lr-exp", -5, -1, step=0.5), # 1e-1 to 1e-5
            # model arch
            "npoints": 100 * 2 ** trial.suggest_int("npoints-exp", 1, 4), # 200 to 3200
            "size-multiplier": 2 ** trial.suggest_int("sm-exp", -1, 3), # 0.5 to 8
            # loss
            "gaussian-sigma": trial.suggest_float("gaussian-sigma", 1, 8, step=0.5) # in meters
        }
        if params["loss"] == "mmd":
            params["mmd-kernel"] = "gaussian"
        elif params["loss"] == "gridmse":
            params["grid-agg"] = trial.suggest_categorical("grid-agg", ["sum", "max"])
        
        if params["output-mode"] == "dense":
            params["out-npoints"] = 2 ** trial.suggest_int("out-npoints-exp", 7, 10) # 128 to 2048
            params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.7, step=0.05)
        elif params["output-mode"] == "seg":
            params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.7, step=0.05)

        # not using TNets

        return params, flags


class pnet2_v1(SearchSpace):

    # defaults are fed in as trial params to the get params function on the 1st trial
    defaults = {
        # main
        "pnet2": True,
        "output-mode": "seg",
        "loss": "gridmse",
        # data
        "subdivide": 5,
        "noise-sigma": 0.1,
        # training
        "optimizer": "adam",
        "batchsize-exp": 4,
        "lr-exp": -1.5,
        # model arch
        "npoints-exp": 3,
        "size-multiplier": 1.0,
        "seg-dropout": 0.2,
        "batchnorm": False,
        # loss
        "gaussian-sigma": 2.5,
    }


    @staticmethod
    def get_params(args, trial):
        flags = [
            "pnet2",
        ]
        params = {
            # main
            "output-mode": trial.suggest_categorical("output-mode", ["dense", "seg"]),
            "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
            # data
            "subdivide": trial.suggest_int("subdivide", 1, 6),
            "noise-sigma": trial.suggest_float("noise-sigma", 0.0, 3.0, step=0.1), # in meters
            # training
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
            "batchsize": 2 ** trial.suggest_int("batchsize-exp", 2, 7), # 4 to 128
            "lr": 10 ** trial.suggest_float("lr-exp", -5, -1, step=0.5), # 1e-1 to 1e-5
            "reducelr-patience": 3,
            # model arch
            "npoints": 100 * 2 ** trial.suggest_int("npoints-exp", 1, 4), # 200 to 3200
            "size-multiplier": trial.suggest_float("size-multiplier", 0.25, 2, step=0.25),
            # loss
            "gaussian-sigma": trial.suggest_float("gaussian-sigma", 1, 8, step=0.5) # in meters
        }
        if params["loss"] == "mmd":
            params["mmd-kernel"] = "gaussian"
        elif params["loss"] == "gridmse":
            params["grid-agg"] = trial.suggest_categorical("grid-agg", ["sum", "max"])
        
        if params["output-mode"] == "dense":
            params["out-npoints"] = 2 ** trial.suggest_int("out-npoints-exp", 7, 10) # 128 to 2048
            params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.7, step=0.05)
        elif params["output-mode"] == "seg":
            params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.7, step=0.05)

        if trial.suggest_categorical("batchnorm", [True, False]):
            flags.append("batchnorm")

        return params, flags






