"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import os
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

# Function definitions
def resize_ims(datapath, height):
    files = []
    datapath = '/home/robert/projects/part_detection/annotated'
    for i in os.listdir(datapath):
        if i.endswith('.jpg'):
            files.append(i)
    files = sorted(files)
    read_ims = []
    for i in files:
        im = imread(datapath + '/'+ i)
        im = resize(im, (height, height * 2))
        read_ims.append(im)

    return read_ims

def draw_line(ims):
    lines_0 = [[(110, 260, 410), (100, 140, 80)], [(160, 165), (115, 95)], [(205, 215), (150, 105)], [(260, 260), (140, 110)], [(315, 305), (140, 100)], [(360, 355), (105, 85)]]
    plt.imshow(ims[0])
    for i in lines_0:
        plt.plot(i[0], i[1], c='yellow')
    plt.show()

    # lines_1 = [[(10, 235, 500), (85, 160, 75)], [(85, 95), (110, 80)], [(140, 170), (200, 100)], [(235, 235), (160, 130)], [(320, 340), (105, 165)], [(400, 410), (75, 100)]]
    # plt.imshow(ims[1])
    # for i in lines_1:
    #     plt.plot(i[0], i[1], c='yellow')
    # plt.show()


def main():
    datapath = '/home/robert/projects/part_detection/annotated'
    read_ims = resize_ims(datapath, 256)
    draw_line(read_ims)

if __name__ == "__main__":
    main()
