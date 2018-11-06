import numpy as np
import pydicom as pd
import h5py
import matplotlib.pyplot as plt

print("Start...")
path = '../data/test/335_20.npy'
sample = np.load(path).item()
file_path = "../../24H_DWI_D3/24H_DWI_D3/335/335/IM-0302-0020.dcm"
ds = pd.read_file(file_path)
arr = np.squeeze(sample['input'], axis=2)
h5f = h5py.File('new.h5', 'w')
h5f.create_dataset('mask', data=arr)
h5f.close()
h5f = h5py.File('new.h5', 'r')
a = h5f['mask']
plt.imshow(arr, cmap=plt.cm.Greys)
plt.show()
print(np.sum(ds.pixel_array))
print(np.max(ds.pixel_array))
print(np.min(ds.pixel_array))
print(ds.pixel_array.shape)
print(arr.shape)
ds.PixelData = arr.tobytes()
ds.Rows = 256
ds.Columns = 256
ds.save_as("new.dcm")
# print(sample['input'])
# print(np.sum(sample['input']))
# print(np.min(sample['input']))
# print(np.max(sample['input']))
# print(sample['output'])
print(np.sum(sample['output']))
print(np.max(sample['output']))
print(np.min(sample['output']))
# print(np.unique(sample['output']))
print("Done")
