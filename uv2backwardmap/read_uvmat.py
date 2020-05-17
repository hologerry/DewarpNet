import sys
import numpy as np
import hdf5storage as h5
np.set_printoptions(threshold=sys.maxsize)

uv_path = 'uvmat/101_6-pp_Page_605-S0H0001.mat'
uv = h5.loadmat(uv_path)['uv']

print(type(uv))  # np.ndarray
print(uv.shape)

first = uv[:, :, 0]
second = uv[:, :, 1]
third = uv[:, :, 2]
# print(first)
# print(second)
# print(third)

print(np.unique(first))   # [0, 1]  mask
print(np.unique(second))  # 0.0 ~ 1.0  x
print(np.unique(third))   # 0.0 ~ 1.0  y
