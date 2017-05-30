import re
import os
import shutil


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

if __name__ == '__main__':
    idx = 0
    rename_given_data_by_label("GIVEN_TRAIN_DATA", "TRAIN")
