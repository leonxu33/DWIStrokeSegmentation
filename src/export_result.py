import numpy as np
import os
import matplotlib.pyplot as plt

result_dir = '../data/result/'
result_file = os.path.join(result_dir, 'output.npy')

result = np.load(result_file)

for i, img in enumerate(result):
    plt.imsave(os.path.join(result_dir, '{}.png'.format(str(i))), np.squeeze(img, axis=2))
