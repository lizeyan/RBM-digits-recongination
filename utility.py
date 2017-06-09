import time
from datetime import datetime
import numba as nb
import os
import numpy as np
from matplotlib import pyplot as plt
import re
from sklearn import metrics
from scipy.ndimage import convolve, gaussian_filter, gaussian_filter1d
from skimage import restoration
from PIL import Image


__last_tic = None
IMAGE_SIZE = (32, 32)
EPS = 1e-10
LABELLED_REGEX = re.compile("\d+\[(?P<digit>\d)\]\.\w+")


def tic():
    global __last_tic
    __last_tic = time.time()


def toc():
    global __last_tic
    print("elapsed time: %f s" % (time.time() - __last_tic))
    __last_tic = None


def log(*args, **kwargs):
    print("[%s]" % str(datetime.now())[:-7], *args, **kwargs)


def to_binary(arr: np.ndarray, axis=-1) -> np.ndarray:
    return np.unpackbits(arr, axis)


def one_hot_encoding(arr: np.ndarray) -> np.ndarray:
    result = np.zeros((np.size(arr), int((np.max(arr) - np.min(arr)) + 1)), dtype=np.float64)
    result[np.arange(np.size(arr)), (arr - np.min(arr)).astype(int)] = 1
    return result


def plot_rbm_features(rbm, output_path: str=None):
    # Plotting
    plt.figure(figsize=(4.2, 4))
    n_components = rbm.n_components
    row = int(np.math.sqrt(n_components))
    column = int((n_components + row - 1) / row)
    for i, comp in enumerate(rbm.components_):
        plt.subplot(row, column, i + 1)
        plt.imshow(comp.reshape(IMAGE_SIZE), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def quantization(arr: np.ndarray, step: int=8) -> np.ndarray:
    return ((arr / step).astype(int) * step).astype(arr.dtype)


def image_preprocess(arr: np.ndarray) -> np.ndarray:
    return restoration.denoise_tv_chambolle(quantization(arr, 4), weight=0.1)
    # return gaussian_filter(arr, sigma=0.5)
    # return restoration.denoise_wavelet(arr)
    # return restoration.denoise_bilateral(arr, sigma_color=0.05, sigma_spatial=15, multichannel=False)
    # return arr


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
            _ = image_preprocess(np.asarray(image))
            # Image.fromarray((_ * 255).astype(np.uint8)).save(os.path.join("PREPROCESSED", filename))
            data_list.append(_)
            label_list.append(int(match.group("digit")))
        else:
            log("Unrecognized filename format: %s" % filename)
    return np.asarray(data_list, np.float64), np.asarray(label_list, np.float64)


def evaluate(predict: np.ndarray, ground_truth: np.ndarray, indicator: str, report=False):
    assert predict.shape == ground_truth.shape
    if np.ndim(predict) is 2:
        accuracy = np.count_nonzero(np.argmax(predict, axis=-1) == np.argmax(ground_truth, axis=-1)) / np.size(predict,
                                                                                                               0)
    else:
        assert np.ndim(predict) is 1
        accuracy = np.count_nonzero(predict.astype(int) == ground_truth.astype(int)) / np.size(predict)
    if report:
        log("%s:\n" % indicator, metrics.classification_report(predict, predict), "accuracy:", accuracy)
    else:
        log("%s:" % indicator, accuracy)


def fit_and_predict(model_class, train_data, train_label, test_data, **kwargs):
    """
    pipeline fit and predict based on sklearn API
    """
    if np.ndim(train_label) is 1:
        model = model_class(**kwargs)
        model.fit(train_data, train_label)
        return model.predict(test_data)
    else:
        assert np.ndim(train_label) is 2
        d = np.size(train_label, 1)
        result = []
        for i in range(d):
            model = model_class(**kwargs)
            model.fit(train_data, train_label[:, i])
            result.append(np.expand_dims(model.predict(test_data), -1))
        return np.concatenate(result, axis=-1)


def min_max_normalize(arr: np.ndarray):
    """
    return a normalized copy of arr.
    @:return (arr - min) / (max - min)
    """
    return (arr - np.min(arr, 0)) / (np.max(arr, 0) - np.min(arr, 0) + EPS)


def nudge_dataset(data, label):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the `size` images in X around by 1px to left, right, down, up
    """
    assert np.ndim(data) is 2
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
    data = np.concatenate([data] + [
        np.apply_along_axis(lambda x, w: convolve(x.reshape(IMAGE_SIZE), mode='constant', weights=w).ravel(), 1, data,
                            vector) for vector in direction_vectors])
    label = np.concatenate([label for _ in range(5)], axis=0)
    return data, label


if __name__ == '__main__':
    tic()
    for i in range(10000):
        log("fuck")
    toc()

