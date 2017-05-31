import numpy as np
from utility import *
import re
import os
from PIL import Image
from sklearn import neural_network, linear_model, pipeline, metrics, ensemble, tree, neighbors
from scipy.ndimage import convolve

EPS = 1e-10
LABELLED_REGEX = re.compile("\d+\[(?P<digit>\d)\]\.\w+")


def read_data(path: str, size):
    """
    read data from the given directory
    :param path: source directory path. All files (not recursively) in this directory will be transversed
    :param size: resize the image to the given size
    :return: two ndarray, data (n, size[0], size[1]) and label (n, )
    """
    data_list = []
    label_list = []
    for filename in sorted(os.listdir(path)):
        match = LABELLED_REGEX.match(filename)
        if match:
            image = Image.open(os.path.join(path, filename)).convert("L").resize(size)
            data_list.append(np.asarray(image))
            label_list.append(int(match.group("digit")))
        else:
            log("Unrecognized filename format: %s" % filename)
    return np.asarray(data_list), np.asarray(label_list)


def evaluate(predict: np.ndarray, ground_truth: np.ndarray, indicator: str, report=False):
    assert predict.shape == ground_truth.shape
    accuracy = np.count_nonzero(predict == ground_truth) / np.size(predict)
    if report:
        log("%s:\n" % indicator, metrics.classification_report(predict, predict), "accuracy:", accuracy)
    else:
        log("%s:" % indicator, accuracy)


def fit_and_predict(model, train_data, train_label, test_data):
    """
    pipeline fit and predict based on sklearn API
    """
    model.fit(train_data, train_label)
    return model.predict(test_data)


def min_max_normalize(arr: np.ndarray):
    """
    return a normalized copy of arr.
    @:return (arr - min) / (max - min)
    """
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + EPS)


def nudge_dataset(data, label, size):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the `size` images in X around by 1px to left, right, down, up
    """
    assert np.ndim(data) is 2 and np.ndim(label) is 1
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]
    data = np.concatenate([data] + [np.apply_along_axis(lambda x, w: convolve(x.reshape(size), mode='constant', weights=w).ravel(), 1, data, vector) for vector in direction_vectors])
    label = np.concatenate([label for _ in range(5)], axis=0)
    return data, label


def main():
    train_data, train_label = read_data("TRAIN", (32, 32))
    test_data, test_label = read_data("TEST", (32, 32))

    train_data = min_max_normalize(train_data)
    test_data = min_max_normalize(test_data)

    flatten_train_data = train_data.reshape(np.size(train_data, 0), -1)
    flatten_test_data = test_data.reshape(np.size(test_data, 0), -1)

    flatten_train_data, train_label = nudge_dataset(flatten_train_data, train_label, train_data.shape[-2:])
    # flatten_binary_train_data = to_binary(flatten_train_data)
    # flatten_binary_test_data = to_binary(flatten_test_data)
    log("train data shape:", flatten_train_data.shape)
    log("test data shape:", flatten_test_data.shape)

    evaluate(fit_and_predict(ensemble.RandomForestClassifier(n_jobs=4, n_estimators=1000, max_depth=38), flatten_train_data, train_label, flatten_test_data), test_label, "random forest")

    # evaluate(fit_and_predict(tree.DecisionTreeClassifier(max_depth=38), flatten_train_data, train_label, flatten_test_data), test_label, "decision tree")

    # evaluate(fit_and_predict(neural_network.MLPClassifier(hidden_layer_sizes=(512, ), verbose=True), flatten_train_data, train_label, flatten_test_data), test_label, "Multi-Layer Perception")

    # evaluate(fit_and_predict(neighbors.KNeighborsClassifier(n_neighbors=1), flatten_train_data, train_label, flatten_test_data), test_label, "knn")

    # evaluate(fit_and_predict(ensemble.AdaBoostClassifier(n_estimators=100), flatten_train_data, train_label, flatten_test_data), test_label, "adaboost")

    evaluate(fit_and_predict(linear_model.SGDClassifier(n_iter=200, shuffle=True, n_jobs=4), flatten_train_data, train_label, flatten_test_data), test_label, "SGD Classifier")

    rbm = neural_network.BernoulliRBM(n_components=1024, learning_rate=0.1, verbose=True, n_iter=10)
    sgd = linear_model.SGDClassifier(n_iter=200, shuffle=True, n_jobs=4)
    for i in range(100):
        evaluate(fit_and_predict(pipeline.Pipeline([("rbm", rbm), ("sgd", sgd)]), flatten_train_data, train_label, flatten_test_data), test_label, "RBM-SGD Classifier")


def to_binary(arr: np.ndarray, axis=-1) -> np.ndarray:
    return np.unpackbits(arr, axis)


if __name__ == '__main__':
    main()
