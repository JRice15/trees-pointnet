import optuna

def pnet1_v1(args, trial):
    flags = []
    params = {
        "output-mode": trial.suggest_categorical("output-mode", ["dense", "seg"]),
        # "output-mode": "dense",
        "loss": trial.suggest_categorical("loss", ["mmd", "gridmse"]),
        "subdivide": trial.suggest_int("subdivide", 1, 5),
        "noise-sigma": trial.suggest_float("noise_sigma", 0.0, 3.0, step=0.1), # in meters
        "optimizer": trial.suggest_categorical("optimizer", ["adam", "adadelta", "nadam", "adamax"]),
        "batchsize": 2 ** trial.suggest_int("batchsize_exp", 3, 7), # 8 to 128
        "lr": 10 ** trial.suggest_float("learning_rate_exp", -5, -1, step=0.5), # 1e-1 to 1e-5
        "npoints": 100 * 2 ** trial.suggest_int("npoints_exp", 1, 4), # 200 to 3200
        "out-npoints": 2 ** trial.suggest_int("out_npoints_exp", 7, 10), # 128 to 2048
        "size-multiplier": 2 ** trial.suggest_int("sm_exp", -1, 3), # 0.5 to 8
        "gaussian-sigma": trial.suggest_int("guassian_sigma", 1, 10, log=True) # in meters
    }
    if params["loss"] == "mmd":
        params["mmd-kernel"] = "gaussian"
        
    if params["output-mode"] == "dense":
        params["dropout"] = trial.suggest_float("dense_dropout", 0.0, 0.6, step=0.05)
    else:
        params["dropout"] = trial.suggest_float("seg_dropout", 0.0, 0.6, step=0.05)

    if trial.suggest_categorical("no_tnet1", [True, False]):
        flags.append("no-tnet1")

    if trial.suggest_categorical("no_tnet2", [True, False]):
        flags.append("no-tnet2")
    else:
        params["ortho-weight"] = 10 ** trial.suggest_int("ortho_exp", -5, 3) # 1e-5 to 1000

    return params, flags




SEARCH_SPACES = {
    "pnet1_v1": pnet1_v1,
}
