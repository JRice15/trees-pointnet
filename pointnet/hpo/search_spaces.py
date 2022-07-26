import optuna

class SearchSpace:
    """
    base class for search spaces
    """

    defaults = None # override this

    @staticmethod
    def get_params(args, trial):
        """
        must return:
            params: dict
            flags: list(str)
        """
        raise NotImplementedError()




class pnet1(SearchSpace):

    # defaults are fed in as trial params to the get_params function on the 1st trial
    defaults = {
        # data
        "subdivide": 8,
        "npoints-exp": 2,
        "noise-sigma": 0.1,
        "handle-small": "fill",
        # training
        "optimizer": "adam",
        "batchsize-exp": 3,
        "lr-exp": -2.5,
        # model arch
        "output-flow": "seg",
        "sm-exp": 0,
        "conf-act": "sigmoid",
        "seg-dropout": 0.3,
        # "batchnorm": False,
        # loss
        "loss": "gridmse",
        "gridmse-agg": "sum",
        "gaussian-sigma": 1.5,
    }


    @staticmethod
    def get_params(args, trial):
        #### PNET 1
        flags = []
        params = {
            # data
            "subdivide": trial.suggest_int("subdivide", 1, 10), # 153.6 to 15.3 meter side length 
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
        if params["loss"] == "gridmse":
            params["gridmse-agg"] = trial.suggest_categorical("gridmse-agg", ["sum", "max"])
        
        # output flow
        if params["output-flow"] == "dense":
            params["out-npoints"] = 2 ** trial.suggest_int("out-npoints-exp", 7, 10) # 128 to 2048
            params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.7, step=0.05)
        elif params["output-flow"] == "seg":
            params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.7, step=0.05)
        
        return params, flags



class pnet2(SearchSpace):

    # defaults are fed in as trial params to the get_params function on the 1st trial
    defaults = {
        # data
        "subdivide": 4,
        "npoints-exp": 3,
        "noise-sigma": 0.5,
        "handle-small": "drop",
        # training
        "optimizer": "adam",
        "batchsize-exp": 4,
        "lr-exp": -2.5,
        # model arch
        "output-flow": "seg",
        "sm-exp": -1,
        "conf-act": "sigmoid",
        "seg-dropout": 0.05,
        "batchnorm": True,
        # loss
        "loss": "gridmse",
        "gridmse-agg": "max",
        "gaussian-sigma": 2.5,
    }


    @staticmethod
    def get_params(args, trial):
        ### PNET 2
        flags = [
            "pnet2",
        ]
        params = {
            # data
            "subdivide": trial.suggest_int("subdivide", 1, 10), # 153.6 to 15.3 meter side length 
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
            "size-multiplier": 2 ** trial.suggest_int("sm-exp", -2, 2), # 0.5 to 4
            "conf-act": trial.suggest_categorical("conf-act", ["relu", "sigmoid"]),
            # loss
            "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
            "gaussian-sigma": trial.suggest_float("gaussian-sigma", 0.5, 6, step=0.5) # in meters
        }
        # losses
        if params["loss"] == "gridmse":
            params["gridmse-agg"] = trial.suggest_categorical("gridmse-agg", ["sum", "max"])
        
        # output flow
        if params["output-flow"] == "dense":
            params["out-npoints"] = 2 ** trial.suggest_int("out-npoints-exp", 7, 10) # 128 to 2048
            params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.7, step=0.05)
        elif params["output-flow"] == "seg":
            params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.7, step=0.05)

        if trial.suggest_categorical("batchnorm", [False, True]):
            flags.append("batchnorm")
        
        return params, flags



class p2p(SearchSpace):

    defaults = {
        "p2p-conf-weight": 0.02,
        "p2p-unmatched-exp": 0,
        "p2p-loc-exp": -1,
    }

    @staticmethod
    def get_params(args, trial):
        flags = []
        params = {
            # data
            "subdivide": 6,
            "npoints": 100 * (2 ** 2),
            "noise-sigma": 0.1,
            "handle-small": "fill",
            # training
            "optimizer": "adam",
            "batchsize": 2 ** 3,
            "lr": 10 ** -2.5,
            # model arch
            "output-flow": trial.suggest_categorical("output-flow", ["seg", "dense"]),
            "size-multiplier": 2 ** 0,
            "conf-act": "sigmoid",
            # loss
            "loss": "p2p",
            "p2p-conf-weight": trial.suggest_float("p2p-conf-weight", 0, 2.0, step=0.01),
            "p2p-unmatched-weight": 10 ** trial.suggest_float("p2p-unmatched-exp", -1, 1, step=0.01),
            "p2p-loc-weight": 10 ** trial.suggest_float("p2p-loc-exp", -2, 1, step=0.01),
        }

        # output flow
        if params["output-flow"] == "dense":
            params["out-npoints"] = trial.suggest_int("out-npoints", 10, 200, step=10)
            params["dropout"] = 0.0
        elif params["output-flow"] == "seg":
            params["dropout"] = 0.3
        
        return params, flags


