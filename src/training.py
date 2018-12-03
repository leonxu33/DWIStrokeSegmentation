import numpy as np
import matplotlib.pyplot as plt
from k_mean import *

data_path = '../data/training/111_16.npy'
data = np.load(data_path).item()
img = np.squeeze(data["input"], axis=2)
mask = np.squeeze(data["output"], axis=2)
# img = data["input"]
print(img.shape)
print(np.max(img))
print(np.min(img))
c_assn, c_cent = k_means(img, 5)

max_centroid = np.argmax(c_cent)
img_seg_output = np.zeros(shape=(c_assn.shape[0], c_assn.shape[1]))
img_mask_output = np.zeros(shape=(c_assn.shape[0], c_assn.shape[1]))
for i in range(img_seg_output.shape[0]):
    for j in range(img_seg_output.shape[1]):
        img_seg_output[i][j] = c_cent[int(c_assn[i][j])]

for i in range(img_mask_output.shape[0]):
    for j in range(img_mask_output.shape[1]):
        if int(c_assn[i][j]) == max_centroid:
            img_mask_output[i][j] = c_cent[int(c_assn[i][j])]

plt.imshow(img_seg_output)
plt.savefig('../data/result/{}'.format("result_seg.png"))
plt.imshow(img_mask_output)
plt.savefig('../data/result/{}'.format("result_mask.png"))
