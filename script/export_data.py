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

# parser = argparse.ArgumentParser()
# parser.add_argument('dcm_path', nargs='?', type=str, default='')
# args = parser.parse_args()
# dcm_dir = args.dcm_path
dcm_dir = '../dataset'

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(training_dir):
    os.mkdir(training_dir)
if not os.path.isdir(valid_dir):
    os.mkdir(valid_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

def main():
    for subdir in os.listdir(dcm_dir):
        if os.path.isdir(os.path.join(dcm_dir, subdir)):
            dir_list.append(subdir)
    dir_list.sort()
    print(dir_list)
    test_set = dir_list[split_num:]
    valid_set = dir_list[split_num * 2:split_num]
    training_set = dir_list[:split_num * 2]
    output_data(test_set, test_dir)
    output_data(valid_set, valid_dir)
    output_data(training_set, training_dir)

def preprocess_data(img_arr, mask=True):
    target_size = (256, 256)
    img_arr = resize(img_arr, target_size)
    if mask:
        img_arr = (img_arr > np.min(img_arr)).astype(int)
    if (img_arr.shape != target_size):
        print("Resize failed...")
    img_arr = np.expand_dims(img_arr, axis=2)
    return img_arr

def output_data(dataset, dir):
    for id in dataset:
        dict_id = {'input': [], 'output': []}
        cleaned_dict = {'input': [], 'output': []}
        for maindir, subdirlist, filelist in os.walk(os.path.join(dcm_dir, id), topdown=False):
            for filename in sorted(filelist):
                if ".dcm" in filename.lower():
                    filepath = os.path.join(maindir, filename)
                    RefDs = pd.read_file(filepath)
                    if "SliceLocation" in RefDs:
                        img_arr = preprocess_data(RefDs.pixel_array, mask=False)
                        dict_id['input'].append(img_arr)
                    else:
                        img_arr = preprocess_data(RefDs.pixel_array, mask=True)
                        dict_id['output'].append(img_arr)
        for i in range(len(dict_id['input'])):
            if np.sum(dict_id['output'][i]) != 0:
                cleaned_dict['input'].append(dict_id['input'][i])
                cleaned_dict['output'].append(dict_id['output'][i])
        print("Loading: {0}, num_sample = ({1}, {2})".format(id, len(cleaned_dict['input']), len(cleaned_dict['output'])))
        for i in range(len(cleaned_dict['input'])):
            dict_i = {'input': cleaned_dict['input'][i], 'output': cleaned_dict['output'][i]}
            np.save(os.path.join(dir, '{0}.{1}'.format(id+'_'+str(i+1), ext)), dict_i)

if __name__ == '__main__':
    main()