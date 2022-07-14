import os

import numpy as np
import pandas as pd
# import torch
from Cython.Includes.numpy import ndarray
from numpy import ndarray
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from pandas.io.parsers import TextFileReader

from skimage.io import imread
from skimage.transform import resize
import torch.utils.data
from typing import Optional, List, Union, Any, Tuple


class WhaleDataset(torch.utils.data.Dataset):
    """Whale dataset."""
    labels: ndarray
    label_ids: ndarray
    names: ndarray

    def __init__(self, data_path: str, mode: str = 'train', height: int = 256, minimum_images: int = 3,
                 alt_data_path: Optional[str] = None) -> None:
        """
        Args:
            data_path (string): path to the dataset
            mode (string): 'train' or 'val'
        """
        self.data_path: str = data_path
        self.alt_data_path: Optional[str] = alt_data_path
        train_data: DataFrame = pd.read_csv(os.path.join(data_path, 'train.csv'))
        unique_labels: ndarray
        unique_label_counts: ndarray
        unique_labels, unique_label_counts = np.unique(train_data['Id'],
                                                       return_counts=True)

        # Remove classes with less than 3 photos
        unique_labels = unique_labels[unique_label_counts >= minimum_images]

        # Remove new_whale
        unique_labels = unique_labels[1:]

        # Create vector of labels and set ids (1 for train, 2 for test)
        self.unique_labels: List[int] = list(unique_labels)
        labels: list[int] = []
        label_ids: list[Union[Union[Series, ExtensionArray, None, ndarray, DataFrame], Any]] = []
        setid: list[int] = []
        names: list[Union[Union[Series, ExtensionArray, None, ndarray, DataFrame], Any]] = []
        unique_labels_seen: ndarray = np.zeros(len(self.unique_labels))
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
        self.mode: str = mode
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
        self.height: int = height

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[ndarray,Any]:
        if self.alt_data_path is not None and os.path.isfile(
                os.path.join(self.alt_data_path, self.names[idx])):
            im: ndarray = imread(os.path.join(self.alt_data_path, self.names[idx]))
            im = np.flip(im, 2)
        else:
            im = imread(
                os.path.join(self.data_path, 'train', self.names[idx]))
        im = resize(im, (self.height * 2, self.height))
        label: Any = self.labels[idx]

        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)

        im = np.float32(np.transpose(im, axes=(2, 0, 1))) / 255

        return im, label


class WhaleTripletDataset(torch.utils.data.Dataset):
    """Whale dataset."""

    def __init__(self, orig_dataset: WhaleDataset, height_list: Optional[List[int]] = None) -> None:
        """
        Args:
            orig_dataset (Dataset): dataset
        """
        if height_list is None:
            height_list = [256, 256, 256]
        self.orig_dataset: WhaleDataset = orig_dataset
        self.height_list: list[int] = height_list

    def __len__(self) -> int:
        return len(self.orig_dataset)

    def __getitem__(self, idx: object) -> Tuple[ndarray,ndarray,int,Any,int]:
        self.orig_dataset.height = self.height_list[0]
        im: ndarray
        lab: Any
        im, lab = self.orig_dataset[idx]
        opts: ndarray = np.where(self.orig_dataset.labels == lab)[0]
        positive_idx: int = opts[np.random.randint(len(opts))]
        opts = np.where(self.orig_dataset.labels != lab)[0]
        negative_idx: int = opts[np.random.randint(len(opts))]
        self.orig_dataset.height = self.height_list[1]
        im_pos: ndarray
        im_pos, _ = self.orig_dataset[positive_idx]
        self.orig_dataset.height = self.height_list[2]
        im_neg, lab_neg = self.orig_dataset[negative_idx]
        return im, im_pos, im_neg, lab, lab_neg
