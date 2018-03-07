import numpy as np
import PIL.Image as Image
import os
import glob
import keras.preprocessing.image as image_generator
from preprocessor import preprocess, extract_green_mask


def augment_data(train_data_path, augmented_data_path, required_examples):

    datagen = image_generator.ImageDataGenerator(width_shift_range=0.2,
                                                 height_shift_range=0.2,
                                                 rotation_range=180,
                                                 zoom_range=0.2,
                                                 horizontal_flip=True,
                                                 vertical_flip=True)

    classes = os.listdir(train_data_path)

    print(classes)

    batch_size = 100
    i = 0

    while required_examples > 0:
        for batch in datagen.flow_from_directory(directory='data/train/',
                                                 batch_size=batch_size,
                                                 save_to_dir=augmented_data_path,
                                                 save_prefix='gen_',
                                                 save_format='png'):
            i += 1
            if i * batch_size > required_examples:
                break

        required_examples -= batch_size


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

    features = np.zeros(shape=(m, image_size * image_size))
    # if grey_scale:
    #     features = np.zeros(shape=(m, image_size * image_size))
    # else:
    #     features = np.zeros(shape=(m, image_size * image_size * 3))

    labels = np.zeros(shape=(m, 1))

    # index of training example
    i = 0
    # iterating through the image files in each category directory
    for category in categories:
        data_files = glob.glob(train_data_path + category + '/*.png')

        # Using RGB data
        # for data_file in data_files:
        #     # loading and preprocessing image with name file
        #     image = preprocess(Image.open(data_file), required_size=image_size)
        #     array = np.asarray(image)
        #     print(array.shape)
        #     array = array[..., :3]
        #
        #     # normalize and enter image data into feature array
        #     features[i, :] = (np.copy(array.flatten()) / 256)
        #     labels[i][0] = category_dict[category]
        #     print('training example %i, file name = %s' % (i, data_file))
        #     i += 1

        for data_file in data_files:
            # loading and extracting green mask from image with name file
            green_mask = extract_green_mask(Image.open(data_file), required_size=image_size)
            print(green_mask.shape)

            # enter green mask into feature array
            features[i, :] = green_mask.flatten()
            labels[i][0] = category_dict[category]
            print('training example %i, file name = %s' % (i, data_file))
            i += 1

    return features, labels, category_dict


def split_data(features, labels, train_fraction, crosval_fraction, test_fraction):
    if (train_fraction + test_fraction + crosval_fraction) != 1.0:
        raise ValueError('fractions do not add to one')

    if np.shape(features)[0] != np.shape(labels)[0]:
        raise ValueError('number of labels and features rows do not match')

    m = np.shape(labels)[0]
    number_of_features = np.shape(features)[1]
    number_of_categories = np.shape(labels)[1]

    # range of possible indices to sample
    available_indices = list(range(0, m))

    train_samples = int(train_fraction * m)
    training_features = np.zeros(shape=(train_samples, number_of_features), dtype='float')
    training_labels = np.zeros(shape=(train_samples, number_of_categories), dtype='int')

    crosval_samples = int(crosval_fraction * m)
    crosval_features = np.zeros(shape=(crosval_samples, number_of_features), dtype='float')
    crosval_labels = np.zeros(shape=(crosval_samples, number_of_categories), dtype='int')

    test_samples = int(test_fraction * m)
    test_features = np.zeros(shape=(test_samples, number_of_features), dtype='float')
    test_labels = np.zeros(shape=(test_samples, number_of_categories), dtype='int')

    training_samples_needed = train_samples
    crosval_samples_needed = crosval_samples
    test_samples_needed = test_samples
    while len(available_indices) > 0:
        random_index = np.random.randint(0, len(available_indices))

        if training_samples_needed > 0:
            training_features[train_samples - training_samples_needed, :] = features[available_indices[random_index], :]
            training_labels[train_samples - training_samples_needed, :] = labels[available_indices[random_index], :]
            training_samples_needed -= 1

        elif crosval_samples_needed > 0:
            crosval_features[crosval_samples - crosval_samples_needed, :] = features[available_indices[random_index], :]
            crosval_labels[crosval_samples - crosval_samples_needed, :] = labels[available_indices[random_index], :]
            crosval_samples_needed -= 1

        elif test_samples_needed > 0:
            test_features[test_samples - test_samples_needed, :] = features[available_indices[random_index], :]
            test_labels[test_samples - test_samples_needed, :] = labels[available_indices[random_index], :]
            test_samples_needed -= 1

        del available_indices[random_index]

    return training_features, training_labels, crosval_features, crosval_labels, test_features, test_labels


def main():
    # features, labels, category_dict = load_train_data('./data/train/')
    #
    # print('features has shape of %s, labels has shape of %s' % (str(np.shape(features)),
    #                                                             str(np.shape(labels))))
    #
    # training_features, training_labels, crosval_features, crosval_labels, test_features, test_labels = \
    #     split_data(features, labels, train_fraction=0.8, crosval_fraction=0.1, test_fraction=0.1)
    #
    # print('training_features has shape of %s, training_labels has shape of %s' % (str(np.shape(training_features)),
    #                                                                               str(np.shape(training_labels))))
    #
    # print('crosval_features has shape of %s, crosval_labels has shape of %s' % (str(np.shape(crosval_features)),
    #                                                                             str(np.shape(crosval_labels))))

    augment_data(train_data_path='data/train/', augmented_data_path='data/augmented_data/', required_examples=1000)


if __name__ == '__main__':
    main()
