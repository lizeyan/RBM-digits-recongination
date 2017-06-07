from matplotlib import pyplot as plt
import numpy as np
from utility import *
import re
import os
from PIL import Image
from sklearn import neural_network, linear_model, pipeline, metrics, ensemble, tree, neighbors, svm
from numba import jit
import keras
from dbn.tensorflow import BinaryRBM


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

    # one_hot_encoding
    one_hot_train_label = one_hot_encoding(train_label)
    one_hot_test_label = one_hot_encoding(test_label)

    log("train data shape:", flatten_train_data.shape)
    log("train label shape:", train_label.shape)
    log("test data shape:", flatten_test_data.shape)
    log("test label shape:", test_label.shape)

    ####################################################################################################################
    # CNN
    ####################################################################################################################
    # cnn = keras.models.Sequential()
    # cnn.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", input_shape=IMAGE_SIZE + (1, )))
    # cnn.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation="relu"))
    # cnn.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    # cnn.add(keras.layers.Dropout(0.25))
    # cnn.add(keras.layers.Flatten())
    # cnn.add(keras.layers.Dense(128, activation="relu"))
    # cnn.add(keras.layers.Dropout(0.5))
    # cnn.add(keras.layers.Dense(10, activation="softmax"))
    # cnn.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
    # cnn.fit(expanded_train_data, one_hot_train_label, batch_size=32, epochs=100, verbose=True, validation_data=(expanded_test_data, one_hot_test_label))
    # evaluate(cnn.predict(expanded_test_data), one_hot_test_label, "CNN")
    # return

    ####################################################################################################################
    # MLP
    ####################################################################################################################
    # mlp = keras.models.Sequential()
    # mlp.add(keras.layers.Dropout(0.25, input_shape=IMAGE_SIZE + (1, )))
    # mlp.add(keras.layers.Flatten())
    # mlp.add(keras.layers.Dense(512, activation="relu"))
    # mlp.add(keras.layers.Dense(256, activation="relu"))
    # mlp.add(keras.layers.Dropout(0.5))
    # mlp.add(keras.layers.Dense(10, activation="softmax"))
    # mlp.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])
    # mlp.fit(expanded_train_data, one_hot_train_label, batch_size=32, epochs=1000, verbose=True, validation_data=(expanded_test_data, one_hot_test_label))
    # evaluate(mlp.predict(expanded_test_data), one_hot_test_label, "MLP")

    ####################################################################################################################
    # Simple and Direct Models
    ####################################################################################################################
    # evaluate(fit_and_predict(linear_model.LogisticRegression, flatten_train_data, train_label, flatten_test_data, C=4.0), test_label, "Logistic Regression", )

    # evaluate(fit_and_predict(linear_model.PassiveAggressiveClassifier, flatten_train_data, train_label, flatten_test_data, C=0.8), test_label, "Passive Aggressor Classifier")

    # evaluate(fit_and_predict(svm.LinearSVR, flatten_train_data, train_label, flatten_test_data, C=1.0, epsilon=0), test_label, "linear SVR")

    # evaluate(fit_and_predict(ensemble.RandomForestClassifier, flatten_train_data, train_label, flatten_test_data, n_jobs=20, n_estimators=100, max_depth=38), test_label, "random forest")

    # evaluate(fit_and_predict(tree.DecisionTreeClassifier, flatten_train_data, train_label, flatten_test_data, max_depth=37), test_label, "decision tree")

    # evaluate(fit_and_predict(neighbors.KNeighborsClassifier, flatten_train_data, train_label, flatten_test_data, n_neighbors=1), test_label, "knn")

    # evaluate(fit_and_predict(ensemble.AdaBoostClassifier, flatten_train_data, train_label, flatten_test_data, n_estimators=100), test_label, "adaboost")

    # evaluate(fit_and_predict(linear_model.SGDClassifier, flatten_train_data, train_label, flatten_test_data, n_iter=200, shuffle=True, n_jobs=20), test_label, "SGD Classifier")

    # evaluate(fit_and_predict(svm.SVC, flatten_train_data, train_label, flatten_test_data, C=8000.0, kernel="rbf", max_iter=-1, cache_size=4096), test_label, "SVC Classifier")

    ####################################################################################################################
    # RBM -> Supervised Models
    ####################################################################################################################
    # transformed_size = (10, 10, 1)
    # rbm = neural_network.BernoulliRBM(n_components=transformed_size[0] * transformed_size[1], learning_rate=0.01, verbose=True, n_iter=100, random_state=0)
    # rbm.fit(flatten_train_data)
    # rbm.fit(flatten_test_data)
    # transformed_train_data = rbm.transform(flatten_train_data)
    # transformed_test_data = rbm.transform(flatten_test_data)
    # evaluate(fit_and_predict(svm.SVC, transformed_train_data, train_label, transformed_test_data, C=512000, cache_size=4096), test_label, "RBM-SVC Classifier")
    # evaluate(fit_and_predict(linear_model.LogisticRegression, transformed_train_data, train_label, transformed_test_data, C=3000.0), test_label, "RBM-Logistic Classifier")

    transformed_size = (8, 8, 1)
    rbm = BinaryRBM(n_hidden_units=transformed_size[0] * transformed_size[1], activation_function="relu", n_epochs=100, batch_size=32, optimization_algorithm="sgd", learning_rate=1e-3)
    rbm.fit(flatten_train_data)
    rbm.fit(flatten_test_data)
    transformed_train_data = rbm.transform(flatten_train_data)
    transformed_test_data = rbm.transform(flatten_test_data)

    ####################################################################################################################
    # RBM->SVC
    ####################################################################################################################
    evaluate(fit_and_predict(svm.SVC, transformed_train_data, train_label, transformed_test_data, C=128000, cache_size=4096), test_label, "RBM-SVC Classifier")

    ####################################################################################################################
    # RBM->Logistic
    ####################################################################################################################
    evaluate(fit_and_predict(linear_model.LogisticRegression, transformed_train_data, train_label, transformed_test_data, C=400.0), test_label, "RBM-Logistic Classifier")

    ####################################################################################################################
    # RBM->RandomForest
    ####################################################################################################################
    evaluate(fit_and_predict(ensemble.RandomForestClassifier, transformed_train_data, train_label, transformed_test_data, n_estimators=800, max_depth=32), test_label, "RBM-random Forest Classifier")

    transformed_train_data = transformed_train_data.reshape((-1, ) + transformed_size)
    transformed_test_data = transformed_test_data.reshape((-1, ) + transformed_size)
    ####################################################################################################################
    # RBM->MLP
    ####################################################################################################################
    # mlp = keras.models.Sequential()
    # mlp.add(keras.layers.Flatten(input_shape=IMAGE_SIZE + (1, )))
    # mlp.add(keras.layers.Dense(32, activation="relu"))
    # mlp.add(keras.layers.Dense(10, activation="softmax"))
    # mlp.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01), metrics=["accuracy"])
    # mlp.fit(transformed_train_data, one_hot_train_label, batch_size=512, epochs=50, verbose=True, validation_data=(transformed_test_data, one_hot_test_label))
    # evaluate(mlp.predict(transformed_test_data), one_hot_test_label, "MLP")

    # model = pipeline.Pipeline([("rbm", rbm), ("supervised", linear_model.SGDClassifier(shuffle=True, n_iter=200, n_jobs=4))])
    # evaluate(fit_and_predict(model, flatten_train_data, train_label, flatten_test_data), test_label, "RBM-PA Classifier 16000")
    # plot_rbm_features(rbm, "output/%d.png" % i)

    # rbm = neural_network.BernoulliRBM(n_components=1024, learning_rate=0.05, verbose=True, n_iter=100)
    # mlp = neural_network.MLPClassifier(hidden_layer_sizes=(512, ), verbose=True)
    # evaluate(fit_and_predict(pipeline.Pipeline([("rbm", rbm), ("mlp", mlp)]), flatten_train_data, train_label, flatten_test_data), test_label, "RBM-SGD Classifier")


if __name__ == '__main__':
    main()
