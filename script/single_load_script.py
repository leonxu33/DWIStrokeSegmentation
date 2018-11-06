import pydicom
import numpy
file_path = "IM-0001-0001.dcm"
 
ds = pydicom.read_file(file_path)
# image
px = ds.pixel_array
numpy.savetxt("px.txt", px)

# other headers such as slice location
ds.SliceLocation

# list all headers information
print(ds)