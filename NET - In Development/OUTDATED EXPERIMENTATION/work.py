import numpy as np, glob
from collections import Counter

c = Counter()
for p in glob.glob("gs_dataset/train/shard_*.npz"):
    z = np.load(p)
    c.update(z["YC"].tolist())
print("polyphony histogram:", dict(sorted(c.items())))