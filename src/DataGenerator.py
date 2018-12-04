import numpy as np
import os

class DataGenerator(object):
    'Generates data for Keras'
    def __init__(self, dim_x = 512, dim_y = 512, dim_z = 6, dim_output = 1,
                batch_size = 2, shuffle = True, verbose = 1,
                scale_data = 1.0, scale_baseline = 1.0):
        'Initialization'
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_output = dim_output
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose
        self.scale_data = scale_data
        self.scale_baseline = scale_baseline

    def generate(self, dir_sample, list_IDs):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_IDs)
            if self.verbose>0:
                print('indexes:', indexes)
            # Generate batches
            imax = int(len(indexes)/self.batch_size)
            if self.verbose>0:
                print('imax:', imax)
            for i in range(imax):
                # Find list of IDs
                list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                if self.verbose>0:
                    print('list_IDs_temp:', list_IDs_temp)
                # Generate data
                X, Y = self.__data_generation(dir_sample, list_IDs_temp)
                if self.verbose>0:
                    print('generated dataset size:', X.shape, Y.shape)

                yield X, Y

    def __get_exploration_order(self, list_IDs):
        'Generates order of exploration'
        # Find exploration order
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
              np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, dir_sample, list_IDs_temp):
        'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        Y = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_output))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store volume
            data_load = np.load(os.path.join(dir_sample, ID)).item()
            X[i, :, :, :] = data_load['seg_img']
            Y[i, :, :, :] = data_load['output']
        X = X * self.scale_data

        # print(X.shape, Y.shape)
        return X, Y