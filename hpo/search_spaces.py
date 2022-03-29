import optuna


def pnet1_v1(args, trial):
    params = {
        "output-mode": trial.suggest_categorical("output-mode", ["dense", "seg"]),
        "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
        "subdivide": trial.suggest_int("subdivide", 1, 5),
        "noise-sigma": trial.suggest_float("noise_sigma", 0.0, 0.1),
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
        "batchsize": 2 ** trial.suggest_int("batchsize_exp", 3, 7), # 8 to 128
        "lr": 10 ** trial.suggest_float("learning_rate_exp", -5, -1, step=0.5), # 1e-1 to 1e-5
        "npoints": 100 * 2 ** trial.suggest_int("npoints_exp", 1, 4), # 200 to 3200
    }
    if params["output-mode"] == "dense":
        params["out-npoints"] = 2 ** trial.suggest_int("out_npoints_exp", 7, 10) # 128 to 2048

    flags = []
    return params, flags




SEARCH_SPACES = {
    "pnet1_v1": pnet1_v1,
}
