
import contextlib
import datetime
import glob
import json
import argparse
import os
from pprint import pprint
from pathlib import PurePath
import time

import h5py
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers
from tensorflow.keras.optimizers import Adam

from src import DATA_DIR, REPO_ROOT, ARGS, patch_generator
from src.losses import get_loss
from src.models import pointnet
from src.tf_utils import MyModelCheckpoint, output_model, load_saved_model
from src.utils import raster_plot, glob_modeldir
# from src.pts_to_gpkg import estimate_pred_thresh, evaluate_preds_to_gpkg, make_gt_gpkg

matplotlib.rc_file_defaults()

def parse_eval_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",required=True,help="name of model to run, with possible timestamp in front")
    parser.parse_args(namespace=ARGS)


def errors_plot(pred, y, results_dir):
    """error plot for predicted vs gt counts"""
    x_w = np.empty(pred.shape)
    x_w.fill(1/pred.shape[0])
    y_w = np.empty(y.shape)
    y_w.fill(1/y.shape[0])
    low = int(min(pred.min(), y.min()))
    high = int(max(pred.max(), y.max()))
    step = max((high - low) // 20, 1)
    bins = range(low, high+1, step)
    plt.hist(y, bins=bins, weights=y_w, label="gt", alpha=0.5, color="green")
    plt.hist(pred, bins=bins, weights=x_w, label="predictions", alpha=0.5, color="blue")
    plt.title("Predictions and Ground Truth Values")
    plt.axvline(np.mean(y), label="gt mean", color="green", linestyle="--")
    plt.axvline(np.mean(pred), label="prediction mean", color="blue", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "preds_vs_gt_hist.png"))
    plt.close()

def count_errors(pred, y):
    if len(pred.shape) > 1:
        pred = K.sum(pred[...,-1], axis=-1)
        y = K.sum(y[...,-1], axis=-1)
    return pred - y


def plot_one_example(x, y, patch_id, outdir, pred=None, naip=None, has_ndvi=False):
    """
    generate raster plots for one example input and output from a dataset
    args:
        x: input from patch generator
        y: targets from patch generator
        patch_id
        outdir: pathlib.PurePath
        pred: prediction from network
        naip: naip image
        has_ndvi: bool, whether x has ndvi as last channel
    """
    patchname = "_".join([str(x) for x in patch_id])
    ylocs = y[y[...,2] == 1][...,:2]

    gt_ntrees = len(ylocs)
    x_locs = x[...,:2]
    x_heights = x[...,2]

    # lidar height (second-highest mode, to avoid noise)
    raster_plot(x_locs, abs_sigma=ARGS.mmd_sigma, weights=x_heights, mode="second-highest",
        filename=outdir.joinpath("{}_lidar_height".format(patchname)), 
        mark=ylocs, zero_one_bounds=False)
    
    # lidar ndvi
    if has_ndvi:
        x_ndvi = x[...,3]
        raster_plot(x_locs, abs_sigma=ARGS.mmd_sigma, weights=x_ndvi, mode="max",
            filename=outdir.joinpath("{}_lidar_ndvi".format(patchname)), 
            mark=ylocs, zero_one_bounds=False)

    if pred is not None:
        # prediction raster
        pred_locs = pred[...,:2]
        pred_weights = pred[...,2]
        raster_plot(pred_locs, abs_sigma=ARGS.mmd_sigma, weights=pred_weights, 
            filename=outdir.joinpath("{}_pred".format(patchname)), 
            mode="sum", mark=ylocs, zero_one_bounds=False)

    if naip is not None:
        plt.imshow(naip[...,:3]) # only use RGB
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(outdir.joinpath(patchname+"_NAIP_RGB.png"))
        plt.clf()
        plt.close()



def pointmatching():
    precisions = []
    recalls = []
    fscores = []
    rmses = []

    ncols = 5
    nrows = len(val_pred)//ncols+1
    fig,ax = plt.subplots(nrows,ncols,figsize=(ncols*10,nrows*10))
    ax = ax.flatten()
    for i in range(len(ax)): ax[i].axis('off')

    row = 0
    col = 0
    for i in range(len(val_pred)):
        gt = val_gt[i]
        pred = cv2.resize(val_pred[i],(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)

        gt_indices = peak_local_max(gt,min_distance=args.min_distance,threshold_rel=args.threshold_rel)
        pred_indices = peak_local_max(pred,min_distance=args.min_distance,threshold_rel=args.threshold_rel)

        # calculate pairwise distances
        dists = pairwise_distances(gt_indices,pred_indices)
        print(len(gt_indices),len(pred_indices),dists.shape)

        # associate each gt tree with all pred trees within radius
        dists[dists>args.max_distance] = np.inf

        # if pred tree associated to multiple gt trees, only take association with smallest distance
        min_inds = np.argmin(dists,axis=0)
        min_dists = np.min(dists,axis=0)
        for j in range(dists.shape[1]):
            dists[:,j] = np.inf
            dists[min_inds[j],j] = min_dists[j]

        # if gt tree associated to multiple pred trees, only take association with smallest distance
        min_inds = np.argmin(dists,axis=1)
        min_dists = np.min(dists,axis=1)
        for j in range(dists.shape[0]):
            dists[j,:] = np.inf
            dists[j,min_inds[j]] = min_dists[j]

        # associated pred trees = true positives
        tp_inds = np.where(np.any(np.logical_not(np.isinf(dists)),axis=0))[0]
        tp = len(tp_inds)

        # un-associated pred trees = false positives
        fp_inds = np.where(np.all(np.isinf(dists),axis=0))[0]
        fp = len(fp_inds)

        # un-associated gt trees = false negatives
        fn_inds = np.where(np.all(np.isinf(dists),axis=1))[0]
        fn = len(fn_inds)

        tp_dists = np.min(dists[:,tp_inds],axis=0)
        rmse = np.sqrt(np.mean(tp_dists**2))

        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        fscore = 2*(precision*recall)/(precision+recall)

        rmses.append(rmse)
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)

        print(f'val {i}:')
        #print('\tdists: ',dists)
        print('\ttp:',tp)
        print('\tfp:',fp)
        print('\tfn:',fn)
        print('\tprecision:',precision)
        print('\trecall:',recall)
        print('\tfscore:',fscore)

        plotargs = {'markersize':10,'markeredgewidth':2}
        ax[i].imshow(val_image[i,:,:,:3])
        ax[i].plot(gt_indices[:,1],gt_indices[:,0],'bo',**plotargs)
        ax[i].plot(pred_indices[tp_inds,1],pred_indices[tp_inds,0],'yx',**plotargs)
        ax[i].plot(pred_indices[fp_inds,1],pred_indices[fp_inds,0],'rx',**plotargs)
        ax[i].legend(['gt','tp','fp'])
        ax[i].axis('off')
    fig.savefig(os.path.join(args.log,'results.png'))

    print('precision:',np.mean(precision))
    print('recall:',np.mean(recall))
    print('fscore:',np.mean(fscore))
    print('rmse:',np.mean(rmse))




def evaluate_model(patchgen, model, model_dir):
    """
    generate predictions from a model on a LidarPatchGen dataset
    args:
        model: Keras Model to predict with
        model_dir: pathlib.PurePath of model's output
    """
    outdir = model_dir.joinpath("results_"+patchgen.name)
    os.makedirs(outdir, exist_ok=True)

    """
    generate predictions
    """
    print("Generating predictions")
    assert patchgen.batch_size == 1
    patchgen.sorted()
    x, y = patchgen.load_all()
    x = np.squeeze(x.numpy())
    y = np.squeeze(y.numpy())
    pred = np.squeeze(model.predict(x))
    patch_ids = patchgen.patch_ids

    """
    Pointmatching
    """
    # denormalize data
    for i in range(len(x)):
        x[i] = patchgen.denormalize_pts(x[i], patch_id=patch_ids[i])
        pred[i] = patchgen.denormalize_pts(pred[i], patch_id=patch_ids[i])
        y[i,:,:2] = patchgen.denormalize_pts(y[i,:,:2], patch_id=patch_ids[i])

    np.savez("test.npz", x=x, pred=pred, gt=y)


    # save raw sample prediction
    with open(outdir.joinpath("sample_predictions.txt"), "w") as f:
        f.write("First 5 predictions, ground truths:\n")
        for i in range(min(5, len(pred))):
            f.write("pred {}:\n".format(i))
            f.write(str(pred[i])+"\n")
            f.write("gt {}:\n".format(i))
            f.write(str(y[i])+"\n")

    """
    data visualizations
    """
    if ARGS.output_mode in ("seg", "dense"):
        print("Generating visualizations...")
        VIS_DIR = outdir.joinpath("visualizations")
        os.makedirs(VIS_DIR, exist_ok=True)

        # grab random 10 examples
        for i in range(0, len(patch_ids), len(patch_ids)//10):
            x_i = x[i]
            y_i = y[i]
            pred_i = pred[i]
            patch_id = patch_ids[i]
            naip = patchgen.get_naip(patch_id)
            plot_one_example(x_i, y_i, patch_id, pred=pred_i, naip=naip, 
                has_ndvi=patchgen.use_ndvi, outdir=VIS_DIR)

    """
    Evaluate Metrics
    """
    print("Evaluating metrics")

    metric_vals = model.evaluate(patchgen)
    if not isinstance(metric_vals, list):
        results = {"loss": metric_vals}
    else:
        results = {model.metrics_names[i]:v for i,v in enumerate(metric_vals)}

    with open(outdir.joinpath("results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults on", patchgen.name, "dataset:")
    for k,v in results.items():
        print(k+":", v)





def main():
    MODEL_DIR = glob_modeldir(ARGS.name)
    MODEL_PATH = MODEL_DIR.joinpath("model.h5")

    # load original params into ARGS object
    params_file = MODEL_DIR.joinpath("params.json")
    with open(params_file, "r") as f:
        params = json.load(f)
    params.pop("name")
    params.pop("test")
    for k,v in params.items():
        setattr(ARGS, k, v)
    ARGS.test = False

    pprint(vars(ARGS))

    DATASET_DIR = DATA_DIR.joinpath("generated/"+ARGS.dsname)

    """
    Evaluation
    """

    model = load_saved_model(MODEL_PATH.as_posix(), ARGS)

    _, val_gen = patch_generator.get_train_val_gens(DATASET_DIR, ARGS.regions, val_split=0.1, test_split=0.1,
                                    val_batchsize=1)
    val_gen.summary()
    evaluate_model(val_gen, model, MODEL_DIR)
    # print("Evaluating with pointmatch")
    # make_gt_gpkg(val_gen)
    # thresh = estimate_pred_thresh(val_gen, MODEL_DIR)
    # results = evaluate_preds_to_gpkg(val_gen, MODEL_DIR, threshold=thresh, save=True)
    # print("Validation pointmatch results:")
    # pprint(results)

    test_gen = patch_generator.get_test_gen(DATASET_DIR, ARGS.regions, val_split=0.1, test_split=0.1)
    test_gen.summary()
    evaluate_model(test_gen, model, MODEL_DIR)
    # print("Evaluating with pointmatch")
    # make_gt_gpkg(test_gen)
    # results = evaluate_preds_to_gpkg(test_gen, MODEL_DIR, threshold=thresh, save=True)
    # print("Test pointmatch results:")
    # pprint(results)




if __name__ == "__main__":
    parse_eval_args()
    main()
