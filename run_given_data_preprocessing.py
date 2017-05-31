import re
import os
import shutil
from run import read_data
import numpy as np
from PIL import Image


def rename_given_data_by_label(path_to_input, path_to_output):
    """
    rename the given data to 'idx[digit].*' format
    """
    def copy(src, digit, image_format):
        global idx
        shutil.copy(src, os.path.join(path_to_output, "%d[%s].%s" % (idx, digit, image_format)))
        idx += 1
    if not os.path.exists(path_to_output):
        os.mkdir(path_to_output)
    for entry in os.listdir(path_to_input):
        full_path = os.path.join(path_to_input, entry)
        if os.path.isfile(full_path):
            match = re.match("(?P<digit>\d)_\d+\..*png", entry)
            if match:
                copy(full_path, match.group("digit"), "png")
                continue
            match = re.match("[\d+]\.(?P<digit>\d).*png", entry)
            if match:
                copy(full_path, match.group("digit"), "png")
                continue
            match = re.match("(?P<digit>\d)\.jpg", entry)
            if match:
                copy(full_path, match.group("digit"), "jpg")
                continue
            match = re.match("(?P<digit>\d)\.jpg.*png", entry)
            if match:
                copy(full_path, match.group("digit"), "png")
                continue
            match = re.match("[\d+]-(?P<digit>\d)\.jpg.*png", entry)
            if match:
                copy(full_path, match.group("digit"), "png")
                continue
            match = re.match("[\d+]-(?P<digit>\d)\.jpg", entry)
            if match:
                copy(full_path, match.group("digit"), "jpg")
                continue
            print(full_path)
        elif os.path.isdir(full_path):
            rename_given_data_by_label(full_path, path_to_output)


def clean_repeated_train_data(path_train, path_test):
    size = (32, 32)
    test_data, _ = read_data(path_test, size)
    for filename in os.listdir(path_train):
        fullpath = os.path.join(path_train, filename)
        arr = np.asarray(Image.open(fullpath).convert("L").resize(size))
        same = np.count_nonzero(np.sum(np.abs(test_data - np.expand_dims(arr, 0)), axis=(-2, -1)) == 0)
        if same != 0:
            os.remove(fullpath)
            print("remove:", fullpath)


if __name__ == '__main__':
    idx = 0
    # rename_given_data_by_label("GIVEN_TRAIN_DATA", "TRAIN")
    # clean_repeated_train_data("TRAIN", "TEST")
