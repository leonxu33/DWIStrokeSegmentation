import numpy as np
import os
import matplotlib.pyplot as plt

training_dir = '../data/training'

temp_dir = '../temp'

shift = 10

for filename in os.listdir(training_dir):
    if filename.endswith(".npy"):
        file_path = os.path.join(training_dir, filename)
        name = filename.split('.')[0]
        print(name)

        data = np.load(file_path).item()
        img = np.squeeze(data['input'], axis=2)
        mask = np.squeeze(data['output'], axis=2)

        img_x = img[::-1, ...]
        mask_x = mask[::-1, ...]

        img_y = img[:, ::-1, ...]
        mask_y = mask[:, ::-1, ...]

        img_xy = np.swapaxes(img, 0, 1)
        mask_xy = np.swapaxes(mask, 0, 1)

        img_sx = np.array(img, copy=True)
        mask_sx = np.array(mask, copy=True)
        img_sx[:-shift, ...] = img_sx[shift:, ...]
        mask_sx[:-shift, ...] = mask_sx[shift:, ...]

        img_sx_ = np.array(img, copy=True)
        mask_sx_ = np.array(mask, copy=True)
        img_sx_[shift:, ...] = img_sx_[:-shift, ...]
        mask_sx_[shift:, ...] = mask_sx_[:-shift, ...]

        img_sy = np.array(img, copy=True)
        mask_sy = np.array(mask, copy=True)
        img_sy[:, :-shift, ...] = img_sy[:, shift:, ...]
        mask_sy[:, :-shift, ...] = mask_sy[:, shift:, ...]

        img_sy_ = np.array(img, copy=True)
        mask_sy_ = np.array(mask, copy=True)
        img_sy_[:, shift:, ...] = img_sy_[:, :-shift, ...]
        mask_sy_[:, shift:, ...] = mask_sy_[:, :-shift, ...]

        dict_x = {'input': np.expand_dims(img_x, axis=2), 'output': np.expand_dims(mask_x, axis=2)}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(name + '_' + 'augflipx', 'npy')), dict_x)

        dict_y = {'input': np.expand_dims(img_y, axis=2), 'output': np.expand_dims(mask_y, axis=2)}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(name + '_' + 'augflipy', 'npy')), dict_y)

        dict_xy = {'input': np.expand_dims(img_xy, axis=2), 'output': np.expand_dims(mask_xy, axis=2)}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(name + '_' + 'augflipxy', 'npy')), dict_xy)

        dict_sx = {'input': np.expand_dims(img_sx, axis=2), 'output': np.expand_dims(mask_sx, axis=2)}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(name + '_' + 'augshiftx', 'npy')), dict_sx)

        dict_sx_ = {'input': np.expand_dims(img_sx_, axis=2), 'output': np.expand_dims(mask_sx_, axis=2)}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(name + '_' + 'augshiftnx', 'npy')), dict_sx_)

        dict_sy = {'input': np.expand_dims(img_sy, axis=2), 'output': np.expand_dims(mask_sy, axis=2)}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(name + '_' + 'augshifty', 'npy')), dict_sy)

        dict_sy_ = {'input': np.expand_dims(img_sx_, axis=2), 'output': np.expand_dims(mask_sy_, axis=2)}
        np.save(os.path.join(training_dir, '{0}.{1}'.format(name + '_' + 'augshiftny', 'npy')), dict_sy_)



# for filename in os.listdir(training_dir):
#     if filename.endswith(".npy"):
#         file_path = os.path.join(training_dir, filename)
#         name = filename.split('.')[0]
#         data = np.load(file_path).item()
#         img = np.squeeze(data['input'], axis=2)
#         mask = np.squeeze(data['output'], axis=2)
#         plt.imsave(os.path.join(training_dir, name + '_img.png'), img)
#         plt.imsave(os.path.join(training_dir, name + '_mask.png'), mask)

# plt.imsave(os.path.join(temp_dir, 'img_x.png'), img_x)
# plt.imsave(os.path.join(temp_dir, 'mask_x.png'), mask_x)
#
# plt.imsave(os.path.join(temp_dir, 'img_y.png'), img_y)
# plt.imsave(os.path.join(temp_dir, 'mask_y.png'), mask_y)
#
# plt.imsave(os.path.join(temp_dir, 'img_xy.png'), img_xy)
# plt.imsave(os.path.join(temp_dir, 'mask_xy.png'), mask_xy)
#
# plt.imsave(os.path.join(temp_dir, 'img_sx.png'), img_sx)
# plt.imsave(os.path.join(temp_dir, 'mask_sx.png'), mask_sx)
#
# plt.imsave(os.path.join(temp_dir, 'img_sx_.png'), img_sx_)
# plt.imsave(os.path.join(temp_dir, 'mask_sx_.png'), mask_sx_)
#
# plt.imsave(os.path.join(temp_dir, 'img_sy.png'), img_sy)
# plt.imsave(os.path.join(temp_dir, 'mask_sy.png'), mask_sy)
#
# plt.imsave(os.path.join(temp_dir, 'img_sy_.png'), img_sy_)
# plt.imsave(os.path.join(temp_dir, 'mask_sy_.png'), mask_sy_)

