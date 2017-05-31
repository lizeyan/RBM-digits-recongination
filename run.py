import numpy as np
from utility import *
import re
import os
from PIL import Image
from sklearn import neural_network, linear_model, pipeline, metrics, ensemble


def read_data(path: str, size):
    data_list = []
    label_list = []
    for filename in os.listdir(path):
        match = LABELLED_REGEX.match(filename)
        if match:
            image = Image.open(os.path.join(path, filename)).convert("L").resize(size)
            data_list.append(np.asarray(image))
            label_list.append(int(match.group("digit")))
        else:
            log("Unrecognized filename format: %s" % filename)
    return np.asarray(data_list), np.asarray(label_list)


def main():

    train_data, train_label = read_data("TRAIN", (32, 32))
    test_data, test_label = read_data("TEST", (32, 32))
    log("train data shape:", train_data.shape)
    log("test data shape:", test_data.shape)

    # rbm = neural_network.BernoulliRBM(n_components=1024, learning_rate=0.06, n_iter=10, random_state=0, verbose=True)
    rf = ensemble.RandomForestClassifier()
    classifier = pipeline.Pipeline(steps=[("rf", rf)])
    classifier.fit(train_data.reshape((np.size(train_data, 0), -1)), train_label)
    predict = classifier.predict(test_data.reshape(np.size(test_data, 0), -1))

    print(metrics.classification_report(test_label, predict))
    accuracy = np.count_nonzero(predict == test_label) / np.size(test_label)
    log("final accuracy:", accuracy)


if __name__ == '__main__':
    LABELLED_REGEX = re.compile("\d+\[(?P<digit>\d)\]\.\w+")
    main()
