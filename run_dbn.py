from dbn.tensorflow import SupervisedDBNClassification
import numpy as np
from utility import *


def main():
    train_data, train_label = read_data("TRAIN", IMAGE_SIZE)
    test_data, test_label = read_data("TEST", IMAGE_SIZE)

    # flat data
    flatten_train_data = train_data.reshape(np.size(train_data, 0), -1)
    flatten_test_data = test_data.reshape(np.size(test_data, 0), -1)

    flatten_train_data, train_label = nudge_dataset(flatten_train_data, train_label)

    # flatten_train_data = np.concatenate([flatten_train_data, gaussian_filter1d(flatten_train_data, sigma=0.5)])
    # train_label = np.concatenate([train_label for _ in range(2)])

    # normalize data
    flatten_train_data = min_max_normalize(flatten_train_data)
    flatten_test_data = min_max_normalize(flatten_test_data)

    expanded_train_data = np.expand_dims(flatten_train_data.reshape((-1,) + IMAGE_SIZE), -1)
    expanded_test_data = np.expand_dims(flatten_test_data.reshape((-1, ) + IMAGE_SIZE), -1)

    dbn = SupervisedDBNClassification(hidden_layers_structure=[256, 256], learning_rate_rbm=0.0005, learning_rate=0.001, n_epochs_rbm=50, n_iter_backprop=100, batch_size=128, activation_function='relu', dropout_p=0.2)
    dbn.fit(flatten_train_data, train_label)
    evaluate(np.asarray(list(dbn.predict(flatten_test_data))), test_label, "DBN")


def example():
    np.random.seed(1337)  # for reproducibility
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.classification import accuracy_score

    from dbn.tensorflow import SupervisedDBNClassification

    # Loading dataset
    digits = load_digits()
    X, Y = digits.data, digits.target

    # Data scaling
    X = (X / 16).astype(np.float32)

    # Splitting data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    # Training
    classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=10,
                                             n_iter_backprop=100,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.fit(X_train, Y_train)

    # Test
    Y_pred = classifier.predict(X_test)
    print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))

if __name__ == '__main__':
    main()
    # example()
