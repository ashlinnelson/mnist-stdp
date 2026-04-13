import numpy as np
import os
from constants import *

print("Generating initial weights ...")

weight_matrix = (np.random.random((n_input, n_e)) + 0.01) * max_weight
weights = weight_matrix.flatten()

os.makedirs('./random/', exist_ok=True)
np.save('./random/initial.npy', weights)

print(f"Success! Saved {len(weights)} connections to ./random/initial.npy")
