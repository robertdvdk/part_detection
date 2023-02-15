import json
import os

import numpy as np
import pandas as pd
# import torch
import skimage.draw
import torchvision.transforms
from numpy import ndarray
from pandas import Series, DataFrame
from pandas.core.arrays import ExtensionArray
from pandas.io.parsers import TextFileReader
import PIL.Image

from skimage.io import imread
import torch.utils.data
from typing import Optional, List, Union, Any, Tuple
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage.transform import resize

# zxhuang
# sys libs
import os
import pickle
import six.moves
import tarfile

# other libs
import numpy as np
import PIL.Image
import torch
import torch.utils.data

class WhaleDataset(torch.utils.data.Dataset):
    """Whale dataset."""
    labels: ndarray
    label_ids: ndarray
    names: ndarray

    def __init__(self, data_path: str, mode: str = 'train', height: int = 256, minimum_images: int = 3, annotated: bool = False,
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
        im = resize(im, (self.height, self.height * 2))
        label: Any = self.labels[idx]

        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)

        im = np.float32(np.transpose(im, axes=(2, 0, 1))) / 255

        return im, label

# def prep_json(train_json):
#     for i,image in enumerate(train_json['images']):
#         pass
#         image['annotations'] = []
#
#     for i,annotation in enumerate(train_json['annotations']):
#         image_id = annotation['image_id']
#         train_json['images'][image_id]['annotations'].append(annotation)



class PartImageNetDataset(torch.utils.data.Dataset):
    """PartImageNet dataset"""

    def __init__(self, data_path: str, mode: str = 'train', height: int = 256, minimum_images: int = 3,
                 alt_data_path: Optional[str] = None) -> None:
        """
        Args:
            data_path (string): path to the dataset
            mode (string): 'train' or 'val'
        """
        self.mode = mode
        self.data_path: str = data_path

        dataset = pd.read_csv(data_path + "/" + "dataset.txt", sep='\t', names=["index", "test", "class", "filename"])

        if mode=="train":
            self.dataset = dataset.loc[dataset['test']==0]
        elif mode=='test':
            self.dataset = dataset.loc[dataset['test']==1]

        # self.alt_data_path: Optional[str] = alt_data_path
        #
        # annFile = os.path.join(data_path, f"{mode}.json")
        #
        # coco = COCO(annFile)
        # self.coco = coco
        #
        # self.unique_label_names = sorted(list({ x['supercategory'] for x in coco.cats.values()    }))
        # coco.imgToLabels = dict()
        # for id,anns in coco.imgToAnns.items():
        #     # start of this img
        #     supercats = []
        #     for ann in anns:
        #         cat_id = ann['category_id']
        #         supercat = coco.cats[cat_id]['supercategory']
        #         supercats.append(supercat)
        #         # break # remove break for sanity check if needed
        #     # print(supercats) # print for sanity check if needed
        #     if not len(set(supercats)) <= 1:
        #         raise RuntimeError("Not all superclasses in this img are the same")
        #     supercat=supercats[0]
        #     coco.imgToLabels[id] = supercat
        #
        # self.label_ids = np.array([y[1] for y in sorted(coco.imgToLabels.items(),key = lambda x: x[0])])
        # self.labels = np.array([self.unique_label_names.index(x) for x in self.label_ids])
        # # self.label_ids = list(range(len(self.labels)))
        # # self.unique_labels=list(set(self.labels))
        #
        # self.height: int = height
        # print("done creating labels")


    def debug_readout(self):
        coco=self.coco

        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))

        imgIds = coco.getImgIds()
        new_img_ids = imgIds[np.random.randint(0, len(imgIds))]
        img = coco.loadImgs(new_img_ids)[0]
        print(img)




        img_filename = img['file_name']
        img_filename_prefix = img_filename.split("_")[0]
        filename = os.path.join(self.data_path,self.mode,img_filename_prefix,img['file_name'])

        I = self.__getitem__(new_img_ids)
        print(I)
        # plt.axis('off')
        plt.imshow(I[0].transpose(1, 2, 0) * 255)

        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        print([ann['category_id'] for ann in anns])


        plt.show()
        I = imread(filename)
        plt.imshow(I)
        coco.showAnns(anns)
        plt.show()
        polygons = []
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg)/2), 2))
                polygons.append(poly)

        for p in polygons:
            mask = skimage.draw.polygon2mask((img['width'], img['height']), p)
            mask = resize(mask, (256, 256))
            plt.imshow(np.where(mask.T, 255, 0))
            plt.show()

    def getmasks(self, idx: int):
        coco = self.coco
        # plt.imshow(I[0].transpose(1, 2, 0) * 255)
        img = coco.loadImgs(idx)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        cat_ids = [ann['category_id'] for ann in anns]
        plt.show()
        polygons = []
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygons.append(poly)
        masks = {'category_id': [], 'mask': []}
        for cat, p in zip(cat_ids, polygons):
            mask = skimage.draw.polygon2mask((img['width'], img['height']), p)
            mask = resize(mask, (256, 256))
            masks['category_id'].append(cat)
            masks['mask'].append(mask)
        return masks

    def __len__(self) -> int:
        print(self.dataset)
        return len(self.dataset['index'])


    def __getitem__(self, idx: int) -> Tuple[ndarray,Any]:
        im = imread(os.path.join())
        # img = self.coco.loadImgs([idx])[0]
        # img_filename = img['file_name']
        # img_filename_prefix = img_filename.split("_")[0]
        # filename = os.path.join(self.data_path,self.mode,img_filename_prefix,img['file_name'])
        #
        # im = imread(filename)
        #
        # im = resize(im, (self.height,self.height), anti_aliasing=True)
        #
        # label: Any = self.labels[idx]
        #
        # if len(im.shape) == 2:
        #     im = np.stack((im,) * 3, axis=-1)
        #
        # im = np.float32(np.transpose(im, axes=(2, 0, 1))) / 255
        # return im, label

class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, split=1, mode = 'train', height: int=256,
                 transform = None, train_samples = None):
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        train_test = pd.read_csv(os.path.join(data_path, 'train_test_split.txt'), delim_whitespace=True, names=['id', 'train'])
        image_names = pd.read_csv(os.path.join(data_path, 'images.txt'), delim_whitespace = True, names=['id', 'filename'])
        labels = pd.read_csv(os.path.join(data_path, 'image_class_labels.txt'), delim_whitespace=True, names=['id', 'label'])
        image_parts = pd.read_csv(os.path.join(data_path, 'parts/part_locs.txt'), delim_whitespace=True, names=['id', 'part_id', 'x', 'y', 'visible'])
        dataset = train_test.merge(image_names, on='id')
        dataset = dataset.merge(labels, on='id')

        if mode == 'train':
            dataset = dataset.loc[dataset['train'] == 1]
            samples = np.arange(len(dataset))
            np.random.shuffle(samples)
            self.trainsamples = samples[:int(len(samples)*split)]
            dataset = dataset.iloc[self.trainsamples]
        elif mode == 'test':
            dataset = dataset.loc[dataset['train'] == 0]
        elif mode == 'val':
            dataset = dataset.loc[dataset['train'] == 1]
            if train_samples is None:
                raise RuntimeError('Please provide the list of training samples'
                                   'to the validation dataset')
            dataset = dataset.drop(dataset.index[train_samples])

        # training images are labelled 1, test images labelled 0. Add these
        # images to the list of image IDs
        self.ids = np.array(dataset['id'])
        self.names = np.array(dataset['filename'])
        # Subtract 1 because classes run from 1-200 instead of 0-199
        self.labels = np.array(dataset['label']) - 1
        parts = {}
        for i in self.ids:
            parts[i] = image_parts[image_parts['id'] == i]
        self.parts = parts


        # self.parts = []
        # with open(f'{data_path}/parts/part_locs.txt', 'r') as fopen:
        #     for line in fopen:
        #         im_id, part_id, x, y, visible = line.split()
        #         if visible == 1:
        #             self.parts.append((part_id, x, y))
        self.height = height

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # imread outputs height x widht
        im = imread(os.path.join(self.data_path, "images", self.names[idx]))
        # im = resize(im, (self.height, self.height))
        label = self.labels[idx]

        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)

        if self.transform:
            im = PIL.Image.fromarray(im)
            im = self.transform(im)

        return im, label

    def get_visible_parts(self, idx: int):
        dataset_id = self.ids[idx]
        parts = self.parts[dataset_id]
        return parts


def pil_loader(path):
    with open(path, 'rb') as f:
        img = PIL.Image.open(f)
        return img.convert('RGB')

class CUB200(torch.utils.data.Dataset):
    """
    CUB200 dataset.
    Variables
    ----------
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _train_data, list of np.array.
        _train_labels, list of int.
        _train_parts, list np.array.
        _train_boxes, list np.array.
        _test_data, list of np.array.
        _test_labels, list of int.
        _test_parts, list np.array.
        _test_boxes, list np.array.
    """
    def __init__(self, root, train=True, transform=None, resize=448):
        """
        Load the dataset.
        Args
        ----------
        root: str
            Root directory of the dataset.
        train: bool
            train/test data split.
        transform: callable
            A function/transform that takes in a PIL.Image and transforms it.
        resize: int
            Length of the shortest of edge of the resized image. Used for transforming landmarks and bounding boxes.
        """
        self._root = root
        self._train = train
        self._transform = transform
        self.loader = pil_loader
        self.newsize = resize
        # 15 key points provided by CUB
        self.num_kps = 15

        if not os.path.isdir(root):
            os.mkdir(root)

        # Load all data into memory for best IO efficiency. This might take a while
        if self._train:
            self._train_data, self._train_labels, self._train_parts, self._train_boxes = self._get_file_list(train=True)
            # assert (len(self._train_data) == 5994
            #         and len(self._train_labels) == 5994)
        else:
            self._test_data, self._test_labels, self._test_parts, self._test_boxes = self._get_file_list(train=False)
            # assert (len(self._test_data) == 5794
            #         and len(self._test_labels) == 5794)

    def __getitem__(self, index):
        """
        Retrieve data samples.
        Args
        ----------
        index: int
            Index of the sample.
        Returns
        ----------
        image: torch.FloatTensor, [3, H, W]
            Image of the given index.
        target: int
            Label of the given index.
        parts: torch.FloatTensor, [15, 4]
            Landmark annotations.
        boxes: torch.FloatTensor, [5, ]
            Bounding box annotations.
        """
        # load the variables according to the current index and split
        if self._train:
            image_path = self._train_data[index]
            target = self._train_labels[index]
            parts = self._train_parts[index]
            boxes = self._train_boxes[index]

        else:
            image_path = self._test_data[index]
            target = self._test_labels[index]
            parts = self._test_parts[index]
            boxes = self._test_boxes[index]

        # load the image
        image = self.loader(image_path)
        image = np.array(image)

        # numpy arrays to pytorch tensors
        parts = torch.from_numpy(parts).float()
        boxes = torch.from_numpy(boxes).float()

        # calculate the resize factor
        # if original image height is larger than width, the real resize factor is based on width
        if image.shape[0] >= image.shape[1]:
            factor = self.newsize / image.shape[1]
        else:
            factor = self.newsize / image.shape[0]

        # transform 15 landmarks according to the new shape
        # each landmark has a 4-element annotation: <landmark_id, column, row, existence>
        for j in range(self.num_kps):

            # step in only when the current landmark exists
            if abs(parts[j][-1]) > 1e-5:
                # calculate the new location according to the new shape
                parts[j][-3] = parts[j][-3] * factor
                parts[j][-2] = parts[j][-2] * factor

        # rescale the annotation of bounding boxes
        # the annotation format of the bounding boxes are <image_id, col of top-left corner, row of top-left corner, width, height>
        boxes[1:] *= factor

        # convert the image into a PIL image for transformation
        image = PIL.Image.fromarray(image)

        # apply transformation
        if self._transform is not None:
            image = self._transform(image)
        return image, target, parts, boxes

    def __len__(self):
        """Return the length of the dataset."""
        if self._train:
            return len(self._train_data)
        return len(self._test_data)

    def _get_file_list(self, train=True):
        """Prepare the data for train/test split and save onto disk."""

        # load the list into numpy arrays
        image_path = self._root + '/CUB_200_2011/images/'
        id2name = np.genfromtxt(self._root + '/CUB_200_2011/images.txt', dtype=str)
        id2train = np.genfromtxt(self._root + '/CUB_200_2011/train_test_split.txt', dtype=int)
        id2part = np.genfromtxt(self._root + '/CUB_200_2011/parts/part_locs.txt', dtype=float)
        id2box = np.genfromtxt(self._root + '/CUB_200_2011/bounding_boxes.txt', dtype=float)

        # creat empty lists
        train_data = []
        train_labels = []
        train_parts = []
        train_boxes = []
        test_data = []
        test_labels = []
        test_parts = []
        test_boxes = []

        # iterating all samples in the whole dataset
        for id_ in range(id2name.shape[0]):
            # load each variable
            image = os.path.join(image_path, id2name[id_, 1])
            # Label starts with 0
            label = int(id2name[id_, 1][:3]) - 1
            parts = id2part[id_*self.num_kps : id_*self.num_kps+self.num_kps][:, 1:]
            boxes = id2box[id_]

            # training split
            if id2train[id_, 1] == 1:
                train_data.append(image)
                train_labels.append(label)
                train_parts.append(parts)
                train_boxes.append(boxes)
            # testing split
            else:
                test_data.append(image)
                test_labels.append(label)
                test_parts.append(parts)
                test_boxes.append(boxes)

        # return accoring to different splits
        if train == True:
            return train_data, train_labels, train_parts, train_boxes
        else:
            return test_data, test_labels, test_parts, test_boxes
if __name__=='__main__':
    pass