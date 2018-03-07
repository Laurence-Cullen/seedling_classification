# import numpy as np
import keras
from keras.models import Sequential
# from keras import optimizers
from keras.layers import Dense
from keras import regularizers
from data_loader import load_train_data, split_data
import matplotlib.pyplot as plt
# import os


def main():
    image_size = 50
    number_of_classes = 12

    # cached_files = os.listdir('cache/')

    # if no cached features and labels exist locally create them and then cache them
    # if 'train_features.csv' not in cached_files or 'train_labels.csv' not in cached_files:

    features, labels, categories = load_train_data(train_data_path='./data/train/',
                                                   image_size=image_size)

    # TODO create a fast caching system
    # np.savetxt('cache/train_features.csv', train_features, delimiter=',', fmt='%.4f')
    # np.savetxt('cache/train_labels.csv', train_labels, delimiter=',', fmt='%i')

    # # if cached features and labels are detected load them into variables
    # else:
    #     train_features = np.genfromtxt('cache/train_features.csv', delimiter=',')
    #     print('training features loaded from cache')
    #     train_labels = np.genfromtxt('cache/train_labels.csv', delimiter=',')
    #     print('training labels loaded from cache')

    binary_training_labels = keras.utils.to_categorical(labels, num_classes=number_of_classes)

    train_features, train_labels, crosval_features, crosval_labels, test_features, test_labels = \
        split_data(features, binary_training_labels, train_fraction=0.9, crosval_fraction=0.0, test_fraction=0.1)

    reg_value = 0.02

    # building nn topology
    model = Sequential()
    model.add(Dense(units=2500,
                    activation='relu',
                    input_dim=image_size ** 2,
                    kernel_regularizer=regularizers.l2(reg_value)))

    model.add(Dense(units=300,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(reg_value)))

    model.add(Dense(units=300,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(reg_value)))

    model.add(Dense(units=number_of_classes,
                    activation='sigmoid',
                    kernel_regularizer=regularizers.l2(reg_value)))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # training_epochs = 200
    # model.fit(train_features, train_labels, epochs=training_epochs, batch_size=100)

    epoch = 0

    # hold historical training and test accuracy
    train_accuracy = {}
    test_accuracy = {}

    try:
        while epoch < 2000:
            model.fit(train_features, train_labels, epochs=1, batch_size=128)
            test_accuracy[epoch] = model.evaluate(test_features, test_labels, batch_size=128)[1]
            train_accuracy[epoch] = model.evaluate(train_features, train_labels, batch_size=128)[1]

            # TODO add sequential model saving

            print('\nepoch = %i\n' % epoch)

            epoch += 1

    except KeyboardInterrupt:
        pass

    # plotting training and test accuracy histories
    plt.plot(train_accuracy.keys(), train_accuracy.values(), label='train')
    plt.plot(test_accuracy.keys(), test_accuracy.values(), label='test')
    axes = plt.gca()
    # axes.set_ylim([0.8, 0.90])
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    test_accuracy = model.evaluate(test_features, test_labels, batch_size=1000)[1]
    print('trained model accuracy on test set = %f' % test_accuracy)


if __name__ == '__main__':
    main()
