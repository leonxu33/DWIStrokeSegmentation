import numpy as np
import os
import argparse
import pydicom as pd
from skimage.transform import *

img_dict = dict()
seg_dict = dict()
data_dir = '../data/'
training_dir = os.path.join(data_dir, 'training/')
test_dir = os.path.join(data_dir, 'test/')
valid_dir = os.path.join(data_dir, 'valid/')
dir_list = []
dcm_dir = ''
ext = 'npy'
split_num = -3

parser = argparse.ArgumentParser()
parser.add_argument('dcm_path', nargs='?', type=str, default='')
args = parser.parse_args()
dcm_dir = args.dcm_path

def preprocess_data(img_arr, mask=True):
    target_size = (256, 256)
    img_arr = resize(img_arr, target_size, anti_aliasing=True)
    if mask:
        img_arr = (img_arr > np.min(img_arr)).astype(int)
    if (img_arr.shape != target_size):
        print("Resize failed...")
    img_arr = np.expand_dims(img_arr, axis=2)
    return img_arr

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(training_dir):
    os.mkdir(training_dir)
if not os.path.isdir(valid_dir):
    os.mkdir(valid_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

for subdir in os.listdir(dcm_dir):
    if os.path.isdir(os.path.join(dcm_dir, subdir)):
        dir_list.append(subdir)
dir_list.sort()
print(dir_list)
test_set = dir_list[split_num:]
valid_set = dir_list[split_num*2:split_num]
training_set = dir_list[:split_num*2]

for id in test_set:
    dict = {'input': [], 'output': []}
    for maindir, subdirlist, filelist in os.walk(os.path.join(dcm_dir, id), topdown=False):
        print("Loading: ", maindir)
        for filename in filelist:
            if ".dcm" in filename.lower():
                filepath = os.path.join(maindir, filename)
                RefDs = pd.read_file(filepath)
                if "SliceLocation" in RefDs:
                    img_arr = preprocess_data(RefDs.pixel_array, mask=False)
                    dict['input'].append(img_arr)
                else:
                    img_arr = preprocess_data(RefDs.pixel_array, mask=True)
                    dict['output'].append(img_arr)
    for i in range(len(dict['input'])):
        dict_i = {'input': dict['input'][i], 'output': dict['output'][i]}
        np.save(os.path.join(test_dir, '{0}.{1}'.format(id+'_'+str(i+1), ext)), dict_i)

for id in valid_set:
    dict = {'input': [], 'output': []}
    for maindir, subdirlist, filelist in os.walk(os.path.join(dcm_dir, id), topdown=False):
        print("Loading: ", maindir)
        for filename in filelist:
            if ".dcm" in filename.lower():
                filepath = os.path.join(maindir, filename)
                RefDs = pd.read_file(filepath)
                img_arr = preprocess_data(RefDs.pixel_array)
                if "SliceLocation" in RefDs:
                    dict['input'].append(img_arr)
                else:
                    dict['output'].append(img_arr)
    for i in range(len(dict['input'])):
        dict_i = {'input': dict['input'][i], 'output': dict['output'][i]}
        np.save(os.path.join(valid_dir, '{0}.{1}'.format(id + '_' + str(i+1), ext)), dict_i)

for id in training_set:
    dict = {'input': [], 'output': []}
    for maindir, subdirlist, filelist in os.walk(os.path.join(dcm_dir, id), topdown=False):
        print("Loading: ", maindir)
        for filename in filelist:
            if ".dcm" in filename.lower():
                filepath = os.path.join(maindir, filename)
                RefDs = pd.read_file(filepath)
                img_arr = preprocess_data(RefDs.pixel_array)
                if "SliceLocation" in RefDs:
                    dict['input'].append(img_arr)
                else:
                    img_arr = np.around(img_arr, decimals=0).astype(int)
                    dict['output'].append(img_arr)
    for i in range(len(dict['input'])):
        dict_i = {'input': dict['input'][i], 'output': dict['output'][i]}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(id + '_' + str(i+1), ext)), dict_i)
