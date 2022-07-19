import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

# max dist that two points can be considered matched, in meters
MAX_MATCH_DIST = 6

def pointmatch(all_gts, all_preds, return_inds=False):
    """
    args:
        all_gts: dict, mapping patchid to array of shape (ntrees,2) where channels are (x,y)
        all_preds: dict, mappin patchid to array of shape (npatches,npoints,3) where channels are (x,y,confidence)
        return_inds: whether to also returns indexes for tp, fp, and fn
    returns:
        dict with keys: precision, recall, fscore, rmse (-1 if no tp), pruned (bool)
        (if return_inds): dict mapping patch_id to dict containing tp, fp, and fn inds
    """
    COST_MATRIX_MAXVAL = 1e10

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tp_dists = []
    pruned = False
    all_inds = {}

    for patch_id in all_gts.keys():

        gt = all_gts[patch_id]
        # get pred, or empty array if missing from dict
        try:
            pred = all_preds[patch_id]
        except KeyError:
            pred = np.empty((0,3))

        if len(gt) == 0:
            all_fp += len(pred)
            if return_inds:
                all_inds[patch_id] = {
                    "tp": np.array([]),
                    "fp": np.arange(len(pred)),
                    "fn": np.array([]),
                }
            continue
        elif len(pred) == 0:
            all_fn += len(gt)
            if return_inds:
                all_inds[patch_id] = {
                    "tp": np.array([]),
                    "fp": np.array([]),
                    "fn": np.arange(len(gt)),
                }
            continue

        # calculate pairwise distances
        dists = pairwise_distances(gt[:,:2],pred[:,:2])
        
        # trees must be within max match distance
        dists[dists>MAX_MATCH_DIST] = np.inf

        # find optimal assignment
        cost_matrix = np.copy(dists)
        cost_matrix[np.isinf(cost_matrix)] = COST_MATRIX_MAXVAL
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        dists[:] = np.inf
        dists[row_ind,col_ind] = cost_matrix[row_ind,col_ind]
        dists[dists>=COST_MATRIX_MAXVAL] = np.inf
        
        # associated pred trees = true positives
        # tp_inds = np.where(np.any(np.logical_not(np.isinf(dists)),axis=0))[0]
        tp_inds = np.where(~np.isinf(dists))[1]
        all_tp += len(tp_inds)

        # un-associated pred trees = false positives
        fp_inds = np.where(np.all(np.isinf(dists),axis=0))[0]
        all_fp += len(fp_inds)

        # un-associated gt trees = false negatives
        fn_inds = np.where(np.all(np.isinf(dists),axis=1))[0]
        all_fn += len(fn_inds)
        
        if len(tp_inds):
            tp_dists = np.min(dists[:,tp_inds],axis=0)
            all_tp_dists.append(tp_dists)

        if return_inds:
            all_inds[patch_id] = {
                "tp": tp_inds,
                "fp": fp_inds,
                "fn": fn_inds,
            }

    if all_tp + all_fp > 0:
        precision = all_tp/(all_tp+all_fp)
    else:
        precision = 0
    if all_tp + all_fn > 0:
        recall = all_tp/(all_tp+all_fn)
    else:
        recall = 0
    if precision + recall > 0:
        fscore = 2*(precision*recall)/(precision+recall)
    else:
        fscore = 0
    if len(all_tp_dists):
        all_tp_dists = np.concatenate(all_tp_dists)
        rmse = np.sqrt(np.mean(all_tp_dists**2))
    else:
        rmse = -1
    
    # calling float/int on a lot of these because json doesn't like numpy dtypes
    results = {
        'pruned': pruned,
        'tp': int(all_tp),
        'fp': int(all_fp),
        'fn': int(all_fn),
        'precision': float(precision),
        'recall': float(recall),
        'fscore': float(fscore),
        'rmse': float(rmse),
    }
    if return_inds:
        return results, all_inds
    return results
