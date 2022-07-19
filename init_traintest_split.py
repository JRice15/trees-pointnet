"""
initialize once a train-test split that is consistent throughout all experiments
"""

import numpy as np
import os
import json

from common.data_handling import get_all_patch_ids, get_all_regions

"""
Constants
"""
VAL_SPLIT = 0.10
TEST_SPLIT = 0.10


if os.path.exists("data/traintest_split.json"):
    raise FileExistsError("Train-test split is already initialized!")





val_step = int(1/VAL_SPLIT)
test_step = int(1/TEST_SPLIT)

train = []
val = []
test = []
# make splits in each region independantly, so we can add or remove a region without 
#   without changing the rest of the data splits
regions = sorted(get_all_regions())
for region in regions:
    patch_ids = sorted(get_all_patch_ids(regions=[region]))
    # random but deterministic shuffle of sorted list
    np.random.default_rng(999).shuffle(patch_ids)
    test += patch_ids[::test_step]
    rest_patch_ids = [x for x in patch_ids if x not in test]
    val += rest_patch_ids[::val_step]
    train += [x for x in rest_patch_ids if x not in val]


data = {
    "train": train,
    "val": val,
    "test": test,
    "train+val": train + val,
    "val_split": VAL_SPLIT,
    "test_split": TEST_SPLIT,
}

with open("data/traintest_split.json", "w") as f:
    json.dump(data, f, indent=1)
