import pydicom as pd
import numpy as np
import os


# path to your dicom images
# i.e. a specific clinical scan sequence such as DWI yields 60 slices of dicom images of size 512 by 512
# then in the DWI images will be stored in a single folder, under which there are 60 "xx.dcm" files
# dcm_path assumes its pointing to that DWI scan folder. AKA we are loading subject by subject

dcm_path = "dcm_folder/"

# create a list of path for each ".dcm"
dcmfilelist = [] 

def takeSecond(elem):
    return elem[1]
	
# go through the path and enlist each file, store the path if the file ends with .dcm
# also store its dcm header's slicelocation so that we can handle them in order
# otherwise dicom images are loaded randomly
for maindir, subdirlist, filelist in os.walk(dcm_path, topdown=False):
    for filename in filelist:
        if ".dcm" in filename.lower():
            filepath = os.path.join(maindir,filename)
            RefDs = pd.read_file(filepath)
            if "SliceLocation" in RefDs:
                dcmfilelist.append((os.path.join(maindir,filename),float(RefDs.SliceLocation)))

# sort output into ascending sequence based on slice location
dcmfilelist_s=sorted(dcmfilelist, key=takeSecond)

# get the (last) dcm image dimension, should be the same for every dcm in the folder
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(dcmfilelist))
# initialize
ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# now load them from top to bottom in descending order (assuming they are axial images)
for (slicename,sliceloc) in dcmfilelist_s:
    ############################ This is the line that read the slice #####################
    ds = pd.read_file(slicename)
    # store the raw image data in the array
    ArrayDicom[:, :, dcmfilelist_s.index((slicename,sliceloc))] = ds.pixel_array

    # now you can visualize with matplotlib, or modify them using numpy
#print(ArrayDicom)
