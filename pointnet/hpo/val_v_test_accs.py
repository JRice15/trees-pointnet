import glob
import argparse
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--name",required=True)
args = parser.parse_args()
name = args.name

dirs = []
val = []
test = []
for path in glob.glob(f"../models/hpo/{name}/{name}_trial*/"):
    test_file = os.path.join(path, "results_test", "results_pointmatch.json")
    val_file = os.path.join(path, "results_validation", "results_pointmatch.json")
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            test_data = json.load(f)
        with open(val_file, "r") as f:
            val_data = json.load(f)
        val.append(val_data["metrics"]["fscore"])
        test.append(test_data["metrics"]["fscore"])
        dirs.append(path)

val = np.array(val)
test = np.array(test)

r2 = r2_score(test, val)

print("val, test max:", val.max(), test.max())
data_max = max(val.max(), test.max())

best = np.argmax(val)
print("best val:", dirs[best])

plt.plot(np.linspace(0, 1.0), np.linspace(0, 1.0), c="black", label="$y=x$", linestyle="dashed", linewidth=1)
plt.scatter(val, test, label="fscore", zorder=3, c="green")
plt.xlabel("Validation set F1-score")
plt.ylabel("Test set F1-score")
plt.xlim(0.0, data_max+0.1)
plt.ylim(0.0, data_max+0.1)
plt.annotate("$y=x$", (data_max+0.01, data_max+0.07), size=8)
plt.annotate("$R^2$={:.3f}".format(r2), (0.02, data_max+0.05))
plt.tight_layout()
plt.savefig(f"studies/{name}/val_v_test_fscore.png")

# zoomed on highest portion
plt.plot(np.linspace(0, 1.0), np.linspace(0, 1.0), c="black", label="$y=x$", linestyle="dashed", linewidth=1)
plt.scatter(val, test, label="fscore", zorder=3, c="green")
plt.xlabel("Validation set F1-score")
plt.ylabel("Test set F1-score")
plt.xlim(0.6, data_max+0.1)
plt.ylim(0.6, data_max+0.1)
plt.tight_layout()
plt.savefig(f"studies/{name}/peak_val_v_test_fscore.png")
