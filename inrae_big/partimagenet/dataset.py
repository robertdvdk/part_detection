import os
import pandas as pd
import skimage.draw
import PIL.Image
from skimage.io import imread
import torch.utils.data
from pycocotools.coco import COCO
import torchvision.transforms as transforms

import numpy as np
import PIL.Image
import torch
import torch.utils.data

class PartImageNetDataset(torch.utils.data.Dataset):
    """PartImageNet dataset"""

    def __init__(self, data_path: str, mode: str = 'train', transform=None,
                 get_masks=False):
        """
        Args:
            data_path (string): path to the dataset
            mode (string): 'train' or 'val'
        """
        self.mode = mode
        self.data_path = data_path
        self.transform = transform
        self.get_masks = get_masks
        dataset = pd.read_csv(data_path + "/" + "newdset.txt", sep='\t', names=["index", "test", "label", "class", "filename"])
        if mode == "train":
            self.dataset = dataset.loc[dataset['test'] == 0]
        elif mode == "val":
            self.dataset = dataset.loc[dataset['test'] == 1]
        elif mode == "test":
            self.dataset = dataset.loc[dataset['test'] == 1]
        annFile = os.path.join(data_path, f"train.json")

        coco = COCO(annFile)
        self.coco = coco

    def getmasks(self, i):
        idx = self.dataset.iloc[i]['index']
        idx = int(idx)
        coco = self.coco
        img = coco.loadImgs(idx)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        cat_ids = [ann['category_id'] for ann in anns]
        polygons = []
        for ann in anns:
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                polygons.append(poly)
        for cat, p in zip(cat_ids, polygons):
            mask = skimage.draw.polygon2mask((img['width'], img['height']), p)
            try:
                mask_tensor[cat] += torch.FloatTensor(mask)
            except NameError:
                mask_tensor = torch.zeros(size=(40, mask.shape[-2], mask.shape[-1]))
                mask_tensor[cat] += torch.FloatTensor(mask)
        try:
            mask_tensor = torch.where(mask_tensor > 0.1, 1, 0).permute(0, 2, 1)
            return mask_tensor
        except UnboundLocalError:
            # if an image has no ground truth parts
            return None

    def __len__(self):
        return len(self.dataset['index'])

    def __getitem__(self, idx):
        curr_row = self.dataset.iloc[idx]
        folder = curr_row['class']
        imgname = curr_row['filename']
        if self.mode == 'train':
            path = f"{self.data_path}/train_train/{folder}/{imgname}"
        elif self.mode == 'test':
            path = f"{self.data_path}/train_test/{folder}/{imgname}"
        im = imread(path)
        label = curr_row['label']
        if len(im.shape) == 2:
            im = np.stack((im,) * 3, axis=-1)
        if self.transform:
            im = PIL.Image.fromarray(im)
            im = self.transform(im)
        if not self.get_masks:
            return im, label, imgname
        mask = self.getmasks(idx)
        if mask == None:
            mask = torch.zeros(size=(40, im.shape[-2], im.shape[-1]))
        mask = transforms.Resize(size=(im.shape[-2], im.shape[-1]),
                interpolation=transforms.InterpolationMode.NEAREST)(mask)
        return im, label, mask

if __name__=='__main__':
    pass