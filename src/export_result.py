import numpy as np
import os
import matplotlib.pyplot as plt

pred_dir = '../data/result/'
pred_file = os.path.join(pred_dir, 'output.npy')
test_dir = '../data/test'

def main():
    pred = np.load(pred_file)
    test_files = sorted([x for x in os.listdir(test_dir) if x.endswith('.npy')])
    print(test_files)
    for i, pred in enumerate(pred):
        test_i_path = os.path.join(test_dir, test_files[i])
        test_i = np.load(test_i_path).item()
        test_i_image = np.squeeze(test_i['input'], axis=2)
        test_i_true = np.squeeze(test_i['output'], axis=2)
        test_i_pred = np.squeeze(pred, axis=2)
        img_true = add_mask_to_image(test_i_image, test_i_true)
        img_pred = add_mask_to_image(test_i_image, np.around(test_i_pred, decimals=0))
        plt.imsave(os.path.join(pred_dir, '{}_pred_origin.png'.format(test_files[i].split('.')[0])), test_i_pred)
        plt.imsave(os.path.join(pred_dir, '{}_true.png'.format(test_files[i].split('.')[0])), img_true)
        plt.imsave(os.path.join(pred_dir, '{}_pred.png'.format(test_files[i].split('.')[0])), img_pred)

def add_mask_to_image(img, mask):
    img_with_mask = np.array(img, copy=True)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i][j] == 1.:
                img_with_mask[i][j] = np.max(img)
    return img_with_mask

if __name__ == '__main__':
    main()