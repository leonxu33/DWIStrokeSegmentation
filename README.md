# Acute Stroke Segmentation from DWI using FCN-8 and U-Net
By Liang Xu, Angela Xu and Katharina Brown  

Stroke is the second-leading cause of death worldwide, but manual detection of acute stroke lesions is a labor-intensive and costly process. This project applies deep learning techniques, specifically two-dimensional U-Net and FCN-8 models, to segmentation of stroke lesions on diffusion-weighted MRI (DWI) images. 

The data used in this project is provided by Stanford Center for Advanced Functional Neuroimaging. 

![alt text](https://github.com/leonxu0910/DWIStrokeSegmentation/blob/master/img/sample_img.png)  

### Packages Required
Pydicom  
Keras  
Tensorflow  
Scikit-image

### How to
#### Data Generation
* To generate data, run ```python export_data.py```.  
* To generate augmented data, run ```python data_aug.py``` after executing previous step.  

#### Training and Testing
* To train model, run ```python stroke_seg_training.py```.
* To test model, run ```python stroke_seg_test.py```.
* Note: To select which model to use (FCN-8 or U-Net), checkout variable ```model``` in the file ```stroke_seg_training.py```. 

#### Result Visualization
* After done testing, run ```python export_result.py``` to output visualization of predicted segmentation. 
