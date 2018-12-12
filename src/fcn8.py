from keras.models import Model
from keras.layers import *
from keras.applications.vgg16 import *
from keras.utils.data_utils import get_file
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from metrics import *
import os

def get_weights_path_vgg16():
    TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',TF_WEIGHTS_PATH,cache_subdir='models')
    #weights_path = '../ckpt/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    return weights_path

def transfer_FCN_Vgg16(input_shape = (224, 224, 1)):
    img_input = Input(shape=input_shape)
    # Block 1
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', data_format="channels_last")(img_input)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', data_format="channels_last")(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1', data_format="channels_last")(conv1_2)

    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', data_format="channels_last")(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', data_format="channels_last")(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2', data_format="channels_last")(conv2_2)

    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1', data_format="channels_last")(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2', data_format="channels_last")(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', data_format="channels_last")(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3', data_format="channels_last")(conv3_3)

    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1', data_format="channels_last")(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2', data_format="channels_last")(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3', data_format="channels_last")(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4', data_format="channels_last")(conv4_3)

    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format="channels_last")(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format="channels_last")(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format="channels_last")(conv5_2)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block5_pool', data_format="channels_last")(conv5_3)

    # Convolutional layers transfered from fully-connected layers
    o = Conv2D(4096, (7, 7), activation='relu', padding='same', name='conv6', data_format="channels_last")(pool5)
    conv7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='conv7', data_format="channels_last")(o)

    ## 4 times upsamping for pool5 layer
    conv7_4 = Conv2DTranspose(2, kernel_size=(4, 4), strides=(4, 4), use_bias=False, data_format="channels_last")(conv7)

    ## 2 times upsampling for pool4
    pool4up = Conv2D(2, (1, 1), activation='relu', padding='same', name="pool4up", data_format="channels_last")(pool4)
    pool4up2 = Conv2DTranspose(2, kernel_size=(2, 2), strides=(2, 2), use_bias=False, data_format="channels_last")(pool4up)
    pool3up = Conv2D(2, (1, 1), activation='relu', padding='same', name="pool3up", data_format="channels_last")(pool3)

    o = Add(name="add")([pool4up2, pool3up, conv7_4])
    o = Conv2DTranspose(1, kernel_size=(8, 8), strides=(8, 8), use_bias=False, data_format="channels_last")(o)

    # Post-processing using 4x4 average pooling before activation
    o = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding="same")(o)
    o = (Activation('sigmoid'))(o)

    model = Model(img_input, o)

    #Create model
    #model = Model(img_input,conv_out)
    weights_path = get_weights_path_vgg16()
    
    # transfer if weights have not been created
    if os.path.isfile(weights_path) == False:
        print("in if statement")
        flattened_layers = model.layers
        index = {}
        for layer in flattened_layers:
            if layer.name:
                index[layer.name]=layer
        vgg16 = VGG16()
        for layer in vgg16.layers:
            weights = layer.get_weights()
            if layer.name == 'fc1':
                weights[0] = np.reshape(weights[0], (7,7,512,4096))
            elif layer.name == 'fc2':
                weights[0] = np.reshape(weights[0], (1,1,4096,4096))
            elif layer.name == 'predictions':
                layer.name = 'predictions_1000'
                weights[0] = np.reshape(weights[0], (1,1,4096,1000))
            if layer.name in index:
                index[layer.name].set_weights(weights)
        model.save_weights(weights_path)
        print( 'Successfully transformed!')
        #else load weights
    else:
        model.load_weights(weights_path, by_name=True)
        print( 'Already transformed!')
    model.compile(loss=dice_coef_loss, optimizer=Adam(lr=0.0002), metrics=[dice_coef])
    return model
