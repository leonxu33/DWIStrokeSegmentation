import pydicom as pd
import numpy as np
import os
import random
import math
from skimage.transform import resize
from skimage.io import imsave

def load_dcm_to_list(dcm_path):
    """
    Load all dcm files from a directory and collect all pixel arrays, braining images and ROI masks, into lists
    :param dcm_path: dcm directory
    :return: lists of ndarray, each array is a pixel array
    """
    img_list = []
    seg_list = []
    for maindir, subdirlist, filelist in os.walk(dcm_path, topdown=False):
        print("Loading: ", maindir)
        for filename in filelist:
            if ".dcm" in filename.lower():
                filepath = os.path.join(maindir, filename)
                RefDs = pd.read_file(filepath)
                if "SliceLocation" in RefDs:
                    img_list.append(RefDs.pixel_array)
                else:
                    seg_list.append(RefDs.pixel_array)
    print("Complete loading...")
    return img_list, seg_list

def preprocess_data(img_list, seg_list):
    """
    Resize each image to the dimension specified.
    :param img_list: list of ndarrays for images
    :param seg_list: list of ndarrays for masks
    :return: lists of ndarray, each array is a pixel array
    """
    img_size = (256, 256)
    for i, arr in enumerate(img_list):
        if arr.shape != img_size:
            print("Resizing img: ", i)
            img_list[i] = resize(arr, img_size)
    for i, arr in enumerate(seg_list):
        if arr.shape != img_size:
            print("Resizing seg: ", i)
            seg_list[i] = resize(arr, img_size)
    print("Finished resizing...")
    return img_list, seg_list

def clean_save_data(dcm_path):
    """
    Load and preprocess dcm images and masks, and save them as .npy files
    :param dcm_path: dcm directory
    :return:
    """
    img_list, seg_list = load_dcm_to_list(dcm_path)
    img, seg = preprocess_data(img_list, seg_list)
    data_dir = '../data/'
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    print("Start saving...")
    np.save('../data/img.npy', img)
    np.save('../data/seg.npy', seg)
    print("Complete saving...")

def get_data():
    """
    Load data from .npy files
    :return:
    """
    img = np.load('../data/img.npy')
    seg = np.load('../data/img.npy')
    return img, seg

# This function is still in progress
def save_data_to_img(img_list, seg_list, directory='../data'):
    img_dir = directory + '/brain/'
    seg_dir = directory + '/mask/'
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    if not os.path.isdir(seg_dir):
        os.mkdir(seg_dir)
    for i, arr in enumerate(img_list):
        print('Saving brain image ', i, '...')
        print(arr)
        file_name = img_dir + str(i) + '.png'
        imsave(file_name, arr)
    for i, arr in enumerate(seg_list):
        print('Saving mask image ', i, '...')
        file_name = seg_dir + str(i) + '.png'
        imsave(file_name, arr)


def split_data(img_list, seg_list, training_ratio=0.98, testing_ratio=0.01, valid_ratio=0.01):
    """
    Split data into training, testing and validation sets
    :param img_list: list of pixel arrays for images
    :param seg_list: list of pixel arrays for masks
    :param training_ratio: percentage of training set
    :param testing_ratio: percentage of testing set
    :param valid_ratio: percentage of validation set
    :return: x(brain image) and y(mask) part of each set of data splitted.
    """
    if len(img_list) != len(seg_list):
        print("ERROR: x and y do not match in length.")
        return
    s = len(img_list)
    s_rand = list(range(s))
    random.shuffle(s_rand)
    testing_index = s_rand[:math.floor(s*testing_ratio)]
    s_rand = s_rand[math.floor(s * testing_ratio):]
    valid_index = s_rand[:math.floor(s*valid_ratio)]
    s_rand = s_rand[math.floor(s * valid_ratio):]
    training_index = s_rand
    # print(len(testing_index))
    # print(len(valid_index))
    # print(len(training_index))
    # print(len(testing_index)+len(valid_index)+len(training_index))
    testing_set_x, testing_set_y, valid_set_x, valid_set_y, training_set_x, training_set_y = [],[],[],[],[],[]
    for index in testing_index:
        testing_set_x.append(img_list[index])
        testing_set_y.append(seg_list[index])
    for index in valid_index:
        valid_set_x.append(img_list[index])
        valid_set_y.append(seg_list[index])
    for index in training_index:
        training_set_x.append(img_list[index])
        training_set_y.append(seg_list[index])
    return training_set_x, training_set_y, testing_set_x, testing_set_y, valid_set_x, valid_set_y

def load_dcm_to_dict(dcm_path):
    """
    Save dictionaries of DWI images and segmentation pixel arrays.
    Each key represents a complete MRI scan and points to an array of pixel arrays.
    :param dcm_path: Load dcm files from selected folder.
    """
    img_dict = dict()
    seg_dict = dict()
    for maindir, subdirlist, filelist in os.walk(dcm_path, topdown=False):
        print("Loading: ", maindir)
        for filename in filelist:
            if ".dcm" in filename.lower():
                filepath = os.path.join(maindir, filename)
                RefDs = pd.read_file(filepath)
                if "SliceLocation" in RefDs:
                    setID = str(RefDs.PatientID)
                    if not (setID in img_dict):
                        img_dict[setID] = np.expand_dims(RefDs.pixel_array, axis=0)
                    else:
                        img_dict[setID] = np.vstack((img_dict[setID], np.expand_dims(RefDs.pixel_array, axis=0)))
                else:
                    setID = str(RefDs.PatientID)
                    if not (setID in seg_dict):
                        seg_dict[setID] = np.expand_dims(RefDs.pixel_array, axis=0)
                    else:
                        seg_dict[setID] = np.vstack((seg_dict[setID], np.expand_dims(RefDs.pixel_array, axis=0)))
    print("Complete loading...")
    return img_dict, seg_dict