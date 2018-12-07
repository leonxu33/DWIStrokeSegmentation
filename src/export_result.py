import numpy as np
import os
import matplotlib.pyplot as plt

result_dir = '../data/result/'
result_file = os.path.join(result_dir, 'output.npy')
test_dir = '../data/test'
result = np.load(result_file)
listfile = [x for x in os.listdir(test_dir) if x.endswith('.npy')]
listfile = sorted(listfile)
print(listfile)
for i, img in enumerate(result):
    plt.imsave(os.path.join(result_dir, '{}.png'.format(listfile[i].split('.')[0])), np.squeeze(img, axis=2))