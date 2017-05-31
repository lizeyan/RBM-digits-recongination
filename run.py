import numpy as np
from utility import *
import re
import os
from PIL import Image
from sklearn import neural_network, linear_model, pipeline, metrics, ensemble, tree, neighbors

LABELLED_REGEX = re.compile("\d+\[(?P<digit>\d)\]\.\w+")


def read_data(path: str, size):
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
    model.fit(train_data, train_label)
    return model.predict(test_data)


def main():
    train_data, train_label = read_data("TRAIN", (32, 32))
    test_data, test_label = read_data("TEST", (32, 32))
    log("train data shape:", train_data.shape)
    log("test data shape:", test_data.shape)
    flatten_train_data = train_data.reshape(np.size(train_data, 0), -1)
    flatten_test_data = test_data.reshape(np.size(test_data, 0), -1)

    rf = ensemble.RandomForestClassifier(n_jobs=4, n_estimators=100)
    evaluate(fit_and_predict(rf, flatten_train_data, train_label, flatten_test_data), test_label, "random forest")

    dt = tree.DecisionTreeClassifier()
    evaluate(fit_and_predict(dt, flatten_train_data, train_label, flatten_test_data), test_label, "decision tree")

    # mlp = neural_network.MLPClassifier()
    # evaluate(fit_and_predict(mlp, flatten_train_data, train_label, flatten_test_data), test_label, "Multi-Layer Perception")

    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    evaluate(fit_and_predict(knn, flatten_train_data, train_label, flatten_test_data), test_label, "knn")

if __name__ == '__main__':
    main()
