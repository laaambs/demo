# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 15:56
# @Author  : lambs
# @File    : preprocess.py
import os
import pickle
import h5py
import json
import numpy as np

split_list = ['train', 'val', 'test']
pkl_filename = 'mini-imagenet-cache-{0}.pkl'
hdf5_filename = 'mini-imagenet-{0}.hdf5'
classes_filename = 'mini-imagenet-{0}-class.json'
label_filename = 'mini-imagenet-{0}-label.json'
mini_imagenet_path = "data/miniimagenet/"


def process_files():
    for split in split_list:
        pkl_file_path = os.path.join(mini_imagenet_path, pkl_filename.format(split))
        if not os.path.isfile(pkl_file_path):
            raise IOError
        # load .pkl file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)
            images, labels_dict = data['image_data'], data['class_dict']
            # test_images: (12000, 84, 84, 3)
            # test_label: a dict of {class_names: indices_list}
            labels = np.array([None] * images.shape[0])
            labels_id = dict()
            num_classes = 0
            for name, indices in labels_dict.items():
                labels_id[name] = num_classes
                labels[indices] = num_classes
                num_classes += 1

        # write images into .hdf5 file
        with h5py.File(os.path.join(mini_imagenet_path, hdf5_filename.format(split)), 'w') as f:
            f.create_dataset("images", data=images)

        # write labels into .json file
        with open(os.path.join(mini_imagenet_path, label_filename.format(split)), 'w') as f:
            json.dump(labels.tolist(), f)

        # write classes into .json file
        with open(os.path.join(mini_imagenet_path, classes_filename.format(split)), 'w') as f:
            json.dump(labels_id, f)
        # remove .pkl file
        if os.path.isfile(pkl_file_path):
            os.remove(pkl_file_path)


if __name__ == "__main__":
    process_files()
