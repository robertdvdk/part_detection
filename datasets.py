import os

import numpy as np
import pandas as pd
# import torch

from skimage.io import imread
from skimage.transform import resize
import torch.utils.data


class WhaleDataset(torch.utils.data.Dataset):
    """Whale dataset."""

    def __init__(self, data_path, mode='train', height=256, minimum_images=3,
                 alt_data_path=None):
        """
        Args:
            data_path (string): path to the dataset
            mode (string): 'train' or 'val'
        """
        self.data_path = data_path
        self.alt_data_path = alt_data_path
        train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        unique_labels, unique_label_counts = np.unique(train_data['Id'],
                                                       return_counts=True)

        # Remove classes with less than 3 photos
        unique_labels = unique_labels[unique_label_counts >= minimum_images]

        # Remove new_whale
        unique_labels = unique_labels[1:]

        # Create vector of labels and set ids (1 for train, 2 for test)
        self.unique_labels = list(unique_labels)
        labels = []
        label_ids = []
        setid = []
        names = []
        unique_labels_seen = np.zeros(len(self.unique_labels))
        for i in range(len(train_data)):
            if train_data['Id'][i] in self.unique_labels:
                labels.append(self.unique_labels.index(train_data['Id'][i]))
                label_ids.append(train_data['Id'][i])
                names.append(train_data['Image'][i])
                if unique_labels_seen[labels[-1]] == 0:
                    setid.append(2)
                else:
                    setid.append(1)
                unique_labels_seen[labels[-1]] += 1
        self.mode = mode
        if mode == 'train':
            self.labels = np.array(labels)[np.array(setid) == 1]
            self.label_ids = np.array(label_ids)[np.array(setid) == 1]
            # self.labels = np.vstack((self.labels*2,self.labels*2+1)).T.reshape(-1)
            self.names = np.array(names)[np.array(setid) == 1]
        if mode == 'val':
            self.labels = np.array(labels)[np.array(setid) == 2]
            self.label_ids = np.array(label_ids)[np.array(setid) == 2]
            self.names = np.array(names)[np.array(setid) == 2]
        if mode == 'no_set':
            self.labels = np.array(labels)
            self.label_ids = np.array(label_ids)
            self.names = np.array(names)
        self.height = height

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.alt_data_path is not None and os.path.isfile(
                os.path.join(self.alt_data_path, self.names[idx])):
            im = imread(os.path.join(self.alt_data_path, self.names[idx]))
            im = np.flip(im, 2)
        else:
            im = imread(
                os.path.join(self.data_path, 'train', self.names[idx]))
        im = resize(im, (self.height * 2, self.height))
        label = self.labels[idx]

        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)

        im = np.float32(np.transpose(im, axes=(2, 0, 1))) / 255

        return im, label


class WhaleTripletDataset(torch.utils.data.Dataset):
    """Whale dataset."""

    def __init__(self, orig_dataset, height_list=None):
        """
        Args:
            orig_dataset (Dataset): dataset
        """
        if height_list is None:
            height_list = [256, 256, 256]
        self.orig_dataset = orig_dataset
        self.height_list = height_list

    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        self.orig_dataset.height = self.height_list[0]
        im, lab = self.orig_dataset[idx]
        opts = np.where(self.orig_dataset.labels == lab)[0]
        positive_idx = opts[np.random.randint(len(opts))]
        opts = np.where(self.orig_dataset.labels != lab)[0]
        negative_idx = opts[np.random.randint(len(opts))]
        self.orig_dataset.height = self.height_list[1]
        im_pos, _ = self.orig_dataset[positive_idx]
        self.orig_dataset.height = self.height_list[2]
        im_neg, lab_neg = self.orig_dataset[negative_idx]
        return im, im_pos, im_neg, lab, lab_neg
