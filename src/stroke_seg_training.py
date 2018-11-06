import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import os
from DataGenerator import *
from unet import *

dir_samples = '../data/'
training_dir = os.path.join(dir_samples, 'training')
valid_dir = os.path.join(dir_samples, 'valid')
ext_data = 'npy'
chkpt_dir = '../ckpt/'
filename_checkpoint = os.path.join(chkpt_dir, 'stroke.ckpt')
batch_size = 32

if not os.path.isdir(chkpt_dir):
    os.mkdir(chkpt_dir)

model = deepEncoderDecoder(num_channel_input = 1,
                        num_channel_output = 1,
                        img_rows = 256,
                        img_cols = 256,
                        lr_init = 0.0002,
                        num_poolings = 3,
                        num_conv_per_pooling = 3,
                        with_bn = True, verbose=1)

params_generator = {'dim_x': 256,
          'dim_y': 256,
          'dim_z': 1,
          'dim_output': 1,
          'batch_size': 4,
          'shuffle': True,
          'verbose': 0,
          'scale_data': 1.0,
          'scale_baseline': 1.0}

list_training_files = [x for x in os.listdir(training_dir) if x.endswith(ext_data)]
list_valid_files = [x for x in os.listdir(valid_dir) if x.endswith(ext_data)]

training_generator = DataGenerator(**params_generator).generate(training_dir, list_training_files)
validation_generator = DataGenerator(**params_generator).generate(valid_dir, list_valid_files)

list_hyper_parameters = [{'lr': 0.0002, 'epochs': 100}]
type_activation_output = 'sigmoid'
if os.path.isfile(filename_checkpoint):
    model.load_weights(filename_checkpoint)
model_checkpoint = ModelCheckpoint(filename_checkpoint, monitor='val_loss', save_best_only=True)

print('Fit generator...')
history = model.fit_generator(
                    generator=training_generator,
                    steps_per_epoch=len(list_training_files)/batch_size,
                    epochs=50,
                    callbacks=[model_checkpoint],
                    validation_data=validation_generator,
                    validation_steps=len(list_valid_files)/batch_size,
                    max_q_size=16)

print(history)