import optuna

class SearchSpaceDefaults:
    """
    namespace for search space default params
    """

    pnet1_v1 = {
        "output-mode": "dense",
        "loss": "mmd",
        "subdivide": 3,
        "noise-sigma": 0.0,
        "optimizer": "adam",
        "batchsize-exp": 5,
        "lr-exp": -3,
        "npoints-exp": 3,
        "out-npoints-exp": 8,
        "sm-exp": 0,
        "gaussian-sigma": 2.5,
        "dense-dropout": 0.0,
        "ortho-weight-exp": -2,
        "no-tnet1": False,
        "no-tnet2": False,
    }

def pnet1_v1(args, trial):
    flags = []
    params = {
        "output-mode": trial.suggest_categorical("output-mode", ["dense", "seg"]),
        "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
        "subdivide": trial.suggest_int("subdivide", 1, 5),
        "noise-sigma": trial.suggest_float("noise-sigma", 0.0, 3.0, step=0.1), # in meters
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
        "batchsize": 2 ** trial.suggest_int("batchsize-exp", 3, 7), # 8 to 128
        "lr": 10 ** trial.suggest_float("lr-exp", -5, -1, step=0.5), # 1e-1 to 1e-5
        "npoints": 100 * 2 ** trial.suggest_int("npoints-exp", 1, 4), # 200 to 3200
        "out-npoints": 2 ** trial.suggest_int("out-npoints-exp", 7, 10), # 128 to 2048
        "size-multiplier": 2 ** trial.suggest_int("sm-exp", -1, 3), # 0.5 to 8
        "gaussian-sigma": trial.suggest_float("guassian-sigma", 1, 8, step=0.5) # in meters
    }
    if params["loss"] == "mmd":
        params["mmd-kernel"] = "gaussian"
        
    if params["output-mode"] == "dense":
        params["dropout"] = trial.suggest_float("dense-dropout", 0.0, 0.6, step=0.05)
    else:
        params["dropout"] = trial.suggest_float("seg-dropout", 0.0, 0.6, step=0.05)

    if trial.suggest_categorical("no-tnet1", [True, False]):
        flags.append("no-tnet1")

    if trial.suggest_categorical("no-tnet2", [True, False]):
        flags.append("no-tnet2")
    else:
        params["ortho-weight"] = 10 ** trial.suggest_int("ortho-exp", -5, 3) # 1e-5 to 1000

    return params, flags




SEARCH_SPACES = {
    "pnet1_v1": pnet1_v1,
}
