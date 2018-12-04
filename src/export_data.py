import numpy as np
import os
import argparse
import pydicom as pd
import matplotlib.pyplot as plt
from skimage.transform import *
from k_mean import *

img_dict = dict()
seg_dict = dict()
data_dir = '../data/'
training_dir = os.path.join(data_dir, 'training/')
test_dir = os.path.join(data_dir, 'test/')
valid_dir = os.path.join(data_dir, 'valid/')
dcm_dir = ''
ext = 'npy'
split_num = -3

# parser = argparse.ArgumentParser()
# parser.add_argument('dcm_path', nargs='?', type=str, default='')
# args = parser.parse_args()
# dcm_dir = args.dcm_path
dcm_dir = '../../24H_DWI_D3/24H_DWI_D3'

invalid_set = ['119', '122', '157', '175', '220', '125', '136', '283', '326']
reversed_set = ['112', '115', '117', '128', '133', '134', '154', '198', '211', '213', '219', '222', '224', '228', '245', '272', '281', '294', '304', '312', '313', '318', '335']

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.isdir(training_dir):
    os.mkdir(training_dir)
if not os.path.isdir(valid_dir):
    os.mkdir(valid_dir)
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

def main():
    dir_list = []
    for subdir in os.listdir(dcm_dir):
        if os.path.isdir(os.path.join(dcm_dir, subdir)):
            dir_list.append(subdir)
    dir_list = list(set(dir_list) - set(invalid_set))
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
        dict_id = {'input': [], 'output': [], 'seg_img': []}
        cleaned_dict = {'input': [], 'output': []}
        for maindir, subdirlist, filelist in os.walk(os.path.join(dcm_dir, id), topdown=False):
            sorted_filelist = sorted(filelist)
            if id in reversed_set:
                image_list = sorted_filelist[0: len(sorted_filelist)//2]
                mask_list = sorted(sorted_filelist[len(sorted_filelist)//2: len(sorted_filelist)], reverse=True)
                sorted_filelist = image_list + mask_list
            for filename in sorted_filelist:
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
            seg_img = run_kmeans(dict_i['input'], 3)
            dict_i['seg_img'] = seg_img
            np.save(os.path.join(dir, '{0}.{1}'.format(id+'_'+str(i+1), ext)), dict_i)
            plt.imsave(os.path.join(dir, '{0}.{1}'.format(id+'_'+str(i+1)+'_img', 'png')), np.squeeze(dict_i['input'], axis=2))
            plt.imsave(os.path.join(dir, '{0}.{1}'.format(id+'_'+str(i+1)+'_mask', 'png')), np.squeeze(dict_i['output'], axis=2))
            plt.imsave(os.path.join(dir, '{0}.{1}'.format(id+'_'+str(i+1)+'_seg_img', 'png')), np.squeeze(dict_i['seg_img'], axis=2))

if __name__ == '__main__':
    main()
