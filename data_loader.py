import pandas as pd
import numpy as np
import PIL.Image as Image
import os
import glob
from preprocessor import preprocess

def load_train_data(train_data_path, image_size=100, grey_scale=False):

    categories = os.listdir(train_data_path)

    # removing osx metadata directory
    if '.DS_Store' in categories:
        categories.remove('.DS_Store')

    category_dict = {}
    m = 0
    for category_index in range(0, len(categories)):
        m += len(glob.glob(train_data_path + categories[category_index] + '/*.png'))
        category_dict[categories[category_index]] = category_index

    if grey_scale:
        features = np.zeros(shape=(m, image_size * image_size))
    else:
        features = np.zeros(shape=(m, image_size * image_size * 3))

    labels = np.zeros(shape=(m, 1))

    # index of training example
    i = 0
    # iterating through the image files in each category directory
    for category in categories:
        file_paths = glob.glob(train_data_path + category + '/*.png')

        for file in file_paths:
            # loading and preprocessing image with name file
            image = preprocess(Image.open(file, mode='rgb'), required_size=image_size)
            array = np.asarray(image).flatten()
            features[i, :] = np.copy(array)
            labels[i][0] = category_dict[category]
            print('training example %i, file name = %s' % (i, file))
            i += 1

def main():
    load_train_data('./data/train/')


if __name__ == '__main__':
    main()