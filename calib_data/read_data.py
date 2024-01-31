import numpy as np

loaded_arrays = np.load("calib_data/MultiMatrix.npz")
print(loaded_arrays.files)
print(loaded_arrays["camMatrix"])