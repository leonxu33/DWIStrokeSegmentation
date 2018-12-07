import numpy as np
import os
from unet import *

img_rows = 224
img_cols = 224
num_poolings = 3
num_conv_per_pooling = 3
lr_init = 0.002
with_batch_norm = True
ratio_validation = 0.1
batch_size = 32
always_retrain = 1
num_epoch = 100
num_channel_input = 1
num_channel_output = 1
filename_checkpoint = '../ckpt/stroke.ckpt'
test_dir = '../data/test/'
result_dir = '../data/result/'

list_test_input = []
list_test_mask = []

for maindir, subdirlist, filelist in os.walk(test_dir, topdown=False):
    for filename in filelist:
        if ".npy" in filename.lower():
            filepath = os.path.join(maindir, filename)
            sample = np.load(filepath).item()
            list_test_input.append(sample['input'])
            list_test_mask.append(sample['output'])

test_input = np.empty((len(list_test_input), img_rows, img_cols, num_channel_input))
test_mask = np.empty((len(list_test_input), img_rows, img_cols, num_channel_output))

for i, arr in enumerate(list_test_input):
    test_input[i, :, :, :] = arr
for i, arr in enumerate(list_test_mask):
    test_mask[i, :, :, :] = arr
# test_input = np.around(test_input, decimals=0).astype(int)
test_mask = np.around(test_mask, decimals=0).astype(int)
print(np.sum(test_input))
print(test_input.shape)
print(test_mask.shape)

model = deepEncoderDecoder(num_channel_input = num_channel_input,
                        num_channel_output = num_channel_output,
                        img_rows = img_rows,
                        img_cols = img_cols,
                        lr_init = lr_init,
                        num_poolings = num_poolings,
                        num_conv_per_pooling = num_conv_per_pooling,
                        with_bn = with_batch_norm, verbose=1)
print('train model:', filename_checkpoint)
print('parameter count:', model.count_params())

model.load_weights(filename_checkpoint)
print('model load from' + filename_checkpoint)

metrics = model.evaluate(test_input, test_mask, batch_size=batch_size)
data_test_output = model.predict(test_input, batch_size=batch_size)
data_test_output_thres = np.around(test_input, decimals=0)

if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

np.save(os.path.join(result_dir, 'output.npy'), data_test_output)
print(metrics)
