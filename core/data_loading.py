import json
import os
import sys
import time
from pathlib import PurePath

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from core import ARGS, REPO_ROOT
from core.viz_utils import raster_plot


class LidarPatchGen(keras.utils.Sequence):
    """
    loads data from a patches h5 file.
    The file should have this structure:

    / [attributes: gridrows, gridcols, gridsize, grid_min_x, grid_min_y]
    /gt: group [attributes: min_trees, max_trees]
    /lidar: group [attributes: min_points, max_points]
    /gt/patchX_Y: dataset, patch from grid column X row Y, shape (numtrees, 2)
        channels: x, y
    /lidar/patchX_Y: dataset, patch from grid column X row Y, shape (numpts, 4)
        channels: x, y, height above ground, ndvi
    /naip/patchX_Y: dataset, square NAIP image from grid column X row Y, shape ~(128,128)
    """

    def __init__(self, patch_ids, name=None, batchsize=None):
        """
        patch_ids: list(tuple), where each tuple is of the format (filename, patch number)
        in the ARGS global object:
            mode
            batchsize
            npoints: points to input per patch
            ndvi: bool, whether to include ndvi channel
            ragged: bool
        """
        assert not (skip_freq is not None and keep_freq is not None)
        self.name = name
        self.batch_size = ARGS.batchsize if batchsize is None else batchsize
        self.use_ndvi = ARGS.ndvi
        self.patch_ids = patch_ids
        self.filenames = list(set([i[0] for i in self.patch_ids]))
        self.files = {f:h5py.File(f, "r") for f in self.filenames}
        self.y_counts_only = False
        if ARGS.mode == "count":
            self.y_counts_only = True
        self.init_rng()
        self.init_data()
        self.batch_time = 0

    def init_rng(self):
        """initialize or reinitialize the random number generatore"""
        self.random = np.random.default_rng(seed=44)

    def init_data(self):
        self.max_trees = self.file["gt"].attrs["max_trees"]
        self.nattributes = 3 + (1 if self.use_ndvi else 0)
        self.npoints = ARGS.npoints
        self.z_max = 50 # for normalization

        self.num_ids = len(self.patch_ids)
        if ARGS.test:
            self.num_ids = 2
        # sort for reproducibility
        self.patch_ids = np.sort(self.patch_ids)
        self.random.shuffle(self.patch_ids)
        if not ARGS.ragged:
            self._x_batch = np.empty((self.batch_size, self.npoints, self.nattributes))
            if self.y_counts_only:
                self._y_batch = np.empty((self.batch_size,))
            else:
                self._y_batch = np.empty((self.batch_size, self.max_trees, 3)) # 3 -> (x,y,isvalid)

    def __len__(self):
        return self.num_ids // self.batch_size

    def __getitem__(self, idx, ):
        return self._ragged_getitem(idx) if ARGS.ragged else self._nonrag_getitem(idx)

    def _nonrag_getitem(self, idx):
        """
        __getitem__ for nonragged outputs
        """
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        self._y_batch.fill(0)
        for i,(filename,patch) in enumerate(self.patch_ids[idx:end_idx]):
            # select <npoints> evenly spaced points randomly from batch.
            #   this is done because indices must be in ascending order, and indexing with an arbitrary array is 
            #   orders of magnitude slower than a simple slice
            x_node = self.files[filename]['lidar/'+patch]
            num_x_pts = x_node.shape[0]
            step = num_x_pts // self.npoints
            leftover = num_x_pts % self.npoints
            if leftover == 0:
                rand_offset = 0
            else:
                rand_offset = self.random.choice(leftover)
            top_offset = leftover - rand_offset
            # get data from file
            self._x_batch[i] = self.file['lidar/'+patch][rand_offset:num_x_pts-top_offset:step, :self.nattributes]

            # normalize data
            # x and y
            min_xy = np.amin(self._x_batch[i,:,:2], axis=0)
            max_xy = np.amax(self._x_batch[i,:,:2], axis=0)
            # z channel
            min_xyz = np.append(min_xy, 0)
            max_xyz = np.append(max_xy, self.z_max)
            if self.use_ndvi:
                # ndvi channel varies from -1 to 1
                min_xyz = np.append(min_xyz, -1)
                max_xyz = np.append(max_xyz, 1)
            self._x_batch[i] = (self._x_batch[i] - min_xyz) / (max_xyz - min_xyz)
            
            # select all gt y points, or just y count
            if self.y_counts_only:
                self._y_batch[i] = self.file['gt/'+patch].shape[0]
            else:
                ydata = self.file['gt/'+patch][:]
                self._y_batch[i,:ydata.shape[0],2] = 1
                self._y_batch[i,ydata.shape[0]:,2] = 0
                self._y_batch[i,:ydata.shape[0],:2] = (ydata - min_xy) / (max_xy - min_xy)

        # shuffle input points within each patch
        #  this shuffles the points within each seperate patch in the same way, but that is random enough for me
        self.random.shuffle(self._x_batch, axis=1)

        x = tf.constant(self._x_batch, dtype=tf.float32)
        y = tf.constant(self._y_batch, dtype=tf.float32)
        self.batch_time += time.perf_counter() - t1
        return x, y

    def _ragged_getitem(self, idx):
        """
        __getitem__ for ragged outputs
        PROBABLY DOESNT WORK ANYMORE, I haven't kept the ragged code up to date
        since it is much slower and less effective
        """
        raise NotImplementedError()
        t1 = time.perf_counter()
        idx = idx * self.batch_size
        end_idx = idx + self.batch_size
        x = []
        y = []
        for i in self.patch_ids[idx:end_idx]:
            x.append(self.file['lidar/'+i][:])
            if self.y_counts_only:
                y.append(self.file['gt/'+i].shape[0])
            else:
                y.append(self.file['gt/'+i][:])
        x = tf.ragged.constant(x, ragged_rank=1, inner_shape=(self.nattributes,), dtype=tf.float32)
        y = tf.constant(y, dtype=tf.float32)
        self.batch_time += time.perf_counter() - t1
        return x, y

    def load_all(self):
        """
        load all examples into one big array. should be used for the validation
        patch generator, because keras doesn't accept a Sequence for val data
        """
        self.sorted()
        x_batches = []
        y_batches = []
        for i in range(len(self)):
            x, y = self[i]
            x_batches.append(x)
            y_batches.append(y)
        x = tf.concat(x_batches, axis=0)
        y = tf.concat(y_batches, axis=0)
        return x, y

    def get_patch(self, i=None, region=None):
        """
        get the full i'th patch of the entire sorted dataset, or from a specific region
        args:
            i: int
            region: str, region name
        returns:
            x, y, patch_id
        """
        # set temporary sorted ids
        old_ids = self.patch_ids
        self.patch_ids = sorted(self.patch_ids)
        old_batchsize = self.batch_size
        self.batch_size = 1
        # get results
        if region is None:
            id = self.patch_ids[i]
            x, y = self[i]
        else:
            id = (region, "patch"+str(i))
            index = self.patch_ids.index(id)
            x, y = self[index]
        # restore correct values
        self.patch_ids = old_ids
        self.batch_size = old_batchsize
        return x[0], y[0], id

    def get_naip(self, patch_id):
        """
        get naip image. Channels: R-G-B-NIR
        args:
            patch_id: tuple: (region name str, patch id str of the form "patch322")
        """
        file, patchname = patch_id
        return self.files[file]["/naip/"+patchname][:]

    def get_batch_shape(self):
        """
        get x,y shape of one batch
        x: (batchsize; npoints or None if ragged; nattributes per point)
        """
        x, y = self[0]
        print("Batch shape x:", x.shape, "y:", y.shape)
        print(x)
        print(y)
        return x.shape, y.shape

    def summary(self):
        print("Dataset", self.name, "from files:")
        for filename in self.filenames:
            print(" ", filename)
        print(" ", self.num_ids, "patches, in", len(self), "batches, batchsize", self.batch_size)
        print(" ", self.npoints, "points per patch.")
        try:
            print("  xbatch shape:", self._x_batch.shape)
            print("  ybatch shape:", self._y_batch.shape)
        except AttributeError:
            pass

    def on_epoch_end(self):
        print("avg batch time:", self.batch_time / len(self))
        print("total batch time:", self.batch_time)
        self.batch_time = 0
        self.random.shuffle(self.patch_ids)

    def sorted(self):
        """put ids in a reproducable order (sorted order)"""
        self.init_rng()
        self.patch_ids = sorted(self.patch_ids)

    def __del__(self):
        try:
            self.file.close()
        except:
            pass # already closed, that's fine

    def evaluate_model(self, model, outdir):
        """
        generate predictions from a model on a LidarPatchGen dataset
        args:
            model: Keras Model to predict with
            outdir: pathlib.PurePath to save output to
        """

        """
        generate predictions
        """
        x, y = self.load_all()
        patch_ids = np.copy(self.patch_ids)
        y = np.squeeze(y.numpy())
        pred = np.squeeze(model.predict(x))
        x = np.squeeze(x.numpy())

        assert len(pred) == len(patch_ids)
        np.savez(outdir.joinpath("predictions.npz").as_posix(), pred=pred, patch_ids=patch_ids)

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
        if ARGS.mode in ["mmd", "pwtt", "pwmmd"]:
            print("Generating visualizations...")
            VIS_DIR = outdir.joinpath("visualizations")
            os.makedirs(VIS_DIR, exist_ok=True)
            # grab random 10 examples

            for i in range(0, 10*5, 5):
                x_i = x[i]
                y_i = y[i]
                pred_i = pred[i]
                patchname = patch_ids[i]
                ylocs = y_i[y_i[...,2] == 1][...,:2]

                gt_ntrees = len(ylocs)
                x_locs = x_i[...,:2]
                x_heights = x_i[...,2]

                # lidar height
                raster_plot(x_locs, gaussian_sigma=ARGS.mmd_sigma, weights=x_heights, mode="max",
                    filename=VIS_DIR.joinpath("{}_lidar_height".format(patchname)), 
                    mark=ylocs, zero_one_bounds=True)
                
                # lidar ndvi
                if ARGS.ndvi:
                    x_ndvi = x_i[...,3]
                    raster_plot(x_locs, gaussian_sigma=ARGS.mmd_sigma, weights=x_ndvi, mode="max",
                        filename=VIS_DIR.joinpath("{}_lidar_ndvi".format(patchname)), 
                        mark=ylocs, zero_one_bounds=True)

                # prediction raster
                pred_locs = pred_i[...,:2]
                pred_weights = pred_i[...,2]
                raster_plot(pred_locs, gaussian_sigma=ARGS.mmd_sigma, weights=pred_weights, 
                    filename=VIS_DIR.joinpath("{}_pred".format(patchname)), 
                    mode="sum", mark=ylocs, zero_one_bounds=True)

                # # top k predicted
                # sorted_preds = pred_i[np.argsort(pred_weights)][::-1]
                # topk_locs = sorted_preds[...,:2][:gt_ntrees]
                # topk_weights = sorted_preds[...,2][:gt_ntrees]
                # raster_plot(topk_locs, gaussian_sigma=ARGS.mmd_sigma, weights=topk_weights, 
                #     filename=VIS_DIR.joinpath("{}_pred_topk".format(patchname)), 
                #     mode="sum", mark=ylocs, zero_one_bounds=True)

                naip = test_gen.get_naip(patchname)
                plt.imshow(naip[...,:3]) # only use RGB
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(VIS_DIR.joinpath(patchname+"_NAIP.png"))
                plt.clf()
                plt.close()

        """
        Evaluate Metrics
        """
        print("Evaluating metrics")

        metric_vals = model.evaluate(test_gen)
        results = {model.metrics_names[i]:v for i,v in enumerate(metric_vals)}

        with open(outdir.joinpath("results.json"), "w") as f:
            json.dump(results, f, indent=2)

        print("\nResults on", self.name, "dataset:")
        for k,v in results.items():
            print(k+":", v)




def get_tvt_split(dsname, regions, val_step, test_step):
    train = []
    val = []
    test = []
    for region in regions:
        regionfile = REPO_ROOT.joinpath("data/generated/{}/{}.h5".format(dsname, region)).as_posix()
        with h5py.File(regionfile, "r") as f:
            patches = sorted(f["lidar"].keys())
        patches = [(region_name, x) for x in patches]
        test += patches[::test_step]
        rest_patches = [x for x in patches if x not in test_patches]
        val += rest_patches[::val_step]
        train += [x for x in rest_patches if x not in val_patches]
    return train, val, test


def get_train_val_gens(dsname, regions, val_split=0.1, test_split=0.1):
    """
    returns:
        train Keras Sequence, val Sequence or raw data, test Sequence
    """
    val_step = int(1/val_split)
    test_step = int(1/val_split)
    train, val, test = get_tvt_split(dsname, regions, val_step, test_step)

    train_gen = LidarPatchGen(train, name="train", skip_freq=val_freq)
    val_gen = LidarPatchGen(val, name="validation", keep_freq=val_freq)

    return train_gen, val_gen

def get_test_gen(val_split=0.1, test_split=0.1):
    val_step = int(1/val_split)
    test_step = int(1/val_split)
    train, val, test = get_tvt_split(dsname, regions, val_step, test_step)

    test_gen = LidarPatchGen(test, name="test", batchsize=1)
    return test_gen



if __name__ == "__main__":
    train_gen, val_gen = get_train_val_gens()
    test_gen = get_test_gen()
    train_gen.summary()
    val_gen.summary()
    test_gen.summary()
