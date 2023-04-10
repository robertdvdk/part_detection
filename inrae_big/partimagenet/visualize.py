# pytorch, vis and image libs
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from PIL import Image
import colorsys
import torch
import torch.nn as nn
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import cv2
import json
from collections import OrderedDict

# sys libs
import os
import argparse
import random

# dataset, utils and model
import sys
import os
sys.path.append(os.path.abspath('../common'))
from dataset import PartImageNetDataset
from torchvision.models import resnet101, ResNet101_Weights
from nets import IndividualLandmarkNet
import skimage

# fix all the randomness for reproducibility
torch.backends.cudnn.enabled = False
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

# number of attributes
num_classes = 110

# arguments
parser = argparse.ArgumentParser(description='Result Visualization')
parser.add_argument('--load', default='', type=str, help='name of model to visualize')
args = parser.parse_args()


def generate_colors(num_colors):
    """
    Generate distinct value by sampling on hls domain.

    Parameters
    ----------
    num_colors: int
        Number of colors to generate.

    Returns
    ----------
    colors_np: np.array, [num_colors, 3]
        Numpy array with rows representing the colors.

    """
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = 0.5
        saturation = 0.9
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors_np = np.array(colors)*255.

    return colors_np

def show_att_on_image(img, mask, output):
    """
    Convert the grayscale attention into heatmap on the image, and save the visualization.

    Parameters
    ----------
    img: np.array, [H, W, 3]
        Original colored image.
    mask: np.array, [H, W]
        Attention map normalized by subtracting min and dividing by max.
    output: str
        Destination image (path) to save.

    Returns
    ----------
    Save the result to output.
    """
    # generate heatmap and normalize into [0, 1]
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # add heatmap onto the image
    merged = heatmap + np.float32(img)

    # re-scale the image
    merged = merged / np.max(merged)
    cv2.imwrite(output, np.uint8(255 * merged))

def landmarks_to_rgb(maps):
    colors = [[0.25, 0, 0.25], [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75],
              [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.75, 0.25, 0],
              [0.75, 0, 0.25], [0, 0.75, 0.25], [0.25, 0.75, 0],
              [0.25, 0, 0.75],
              [0, 0.25, 0.75], [0.75, 0.5, 0], [0.75, 0, 0.5], [0, 0.75, 0.5],
              [0.5, 0.75, 0], [0.5, 0, 0.75], [0, 0.5, 0.75],
              [0.75, 0.5, 0.25],
              [0.25, 0.75, 0.5], [0.5, 0.25, 0.75], [0.5, 0.75, 0.25],
              [0.75, 0.25, 0.5], [0.25, 0.5, 0.75], [0.75, 0, 0.75]]
    rgb = np.ones((maps.shape[0],maps.shape[1],3))
    for c in range(3):
        for h in range(len(maps)):
            for w in range(len(maps)):
                rgb[h, w, c] += colors[maps[h, w] % 25][c] * 2

    return rgb

def plot_assignment(root, assign_hard, num_parts):
    """
    Blend the original image and the colored assignment maps.

    Parameters
    ----------
    root: str
        Root path for saving visualization results.
    assign_hard: np.array, [H, W]
        Hard assignment map (int) denoting the deterministic assignment of each pixel. Generated via argmax.
    num_parts: int, number of object parts.

    Returns
    ----------
    Save the result to root/assignment.png.

    """
    maps = np.eye(num_parts, dtype='uint8')[assign_hard]
    maps = np.expand_dims(np.transpose(maps, (2, 0, 1)), axis=0)
    grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[3]),
                                    torch.arange(maps.shape[2]))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    map_sums = maps.sum(3).sum(2)
    maps_x = grid_x * maps
    maps_y = grid_y * maps
    loc_x = maps_x.sum(3).sum(2) / map_sums
    loc_y = maps_y.sum(3).sum(2) / map_sums
    colors = [[0.25, 0, 0.25], [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75],
          [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.75, 0.25, 0],
          [0.75, 0, 0.25], [0, 0.75, 0.25], [0.25, 0.75, 0], [0.25, 0, 0.75],
          [0, 0.25, 0.75], [0.75, 0.5, 0], [0.75, 0, 0.5], [0, 0.75, 0.5],
          [0.5, 0.75, 0], [0.5, 0, 0.75], [0, 0.5, 0.75], [0.75, 0.5, 0.25],
          [0.25, 0.75, 0.5], [0.5, 0.25, 0.75], [0.5, 0.75, 0.25],
          [0.75, 0.25, 0.5], [0.25, 0.5, 0.75]]
    fig, ax = plt.subplots(1, 1)
    i = 0
    im = Image.open(os.path.join(root, 'input.png')).convert('RGB')
    im = transforms.ToTensor()(im)
    landmarks = landmarks_to_rgb(assign_hard[:, :])
    ax.imshow((skimage.transform.resize(landmarks, (256, 256)) * 0.4 + skimage.transform.resize((im[:, :, :].permute(1, 2, 0).numpy()), (256, 256))) * 0.6)
    x_coords = loc_y[i, :].detach().cpu()*256/maps.shape[-1]
    y_coords = loc_x[i, :].detach().cpu() * 256 / maps.shape[-1]
    cols = colors[0:loc_x.shape[1]]
    n = np.arange(loc_x.shape[1] + 1)
    for xi, yi, col_i, mark in zip(x_coords, y_coords, cols, n):
        if np.where(assign_hard == mark, 1, 0).sum() == 0:
            continue
        ax.scatter(xi, yi, color=col_i, marker=f'${mark}$')
    plt.savefig(os.path.join(root, 'assignment.png'))

def main():
    height=256
    # define data transformation (no crop)
    test_transforms = transforms.Compose([
        transforms.Resize(size=height),
        transforms.ToTensor(),
        ])

    # define test dataset and loader
    dataset = PartImageNetDataset(data_path='../datasets/pim', mode='test', transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=False)

    # create a dataloader iter instance
    test_loader_iter = iter(test_loader)

    # define the figure layout
    fig_rows = 5
    fig_cols = 5
    f_assign, axarr_assign = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols*2,fig_rows*2))
    f_assign.subplots_adjust(wspace=0, hspace=0)

    # load the model in eval mode
    # with batch size = 1, we only support single GPU visaulization
    num_landmarks = 50
    model_name_init = 'PIM_50parts_all_losses_normal_ABLATION_INDIVIDUAL.pt'
    weights = ResNet101_Weights.DEFAULT
    basenet = resnet101(weights=weights)
    net = IndividualLandmarkNet(basenet, num_landmarks, num_classes=num_classes, height=height)
    net.load_state_dict(torch.load(model_name_init), strict=False)

    # load model
    net.eval().cuda()

    with torch.no_grad():
        # the visualization code
        current_id = 0
        for col_id in range(fig_cols):
            for j in range(fig_rows):

                # inference the model
                input, target, path = next(test_loader_iter)
                img_path = path[0].split('/')[-1].split('.')[0]
                input = input.cuda()
                target = target.cuda()
                current_id += 1
                with torch.no_grad():
                    print("Visualizing %dth image..." % current_id)
                    _, assign, _, _, output = net(input)

                # define root for saving results and make directories correspondingly
                root = os.path.join('./visualization', args.load, f'{current_id}_{img_path}_{num_landmarks}parts')
                os.makedirs(root, exist_ok=True)
                os.makedirs(os.path.join(root, 'assignments'), exist_ok=True)
                print(assign.shape)
                # denormalize the image and save the input
                save_input = torch.nn.functional.interpolate(input.data[0].unsqueeze(0), size=(height, height), mode='bilinear', align_corners=False).squeeze(0)
                img = torchvision.transforms.ToPILImage()(save_input)
                img.save(os.path.join(root, 'input.png'))

                # save the labels and pred
                label = target.data.item()
                _, prediction = torch.max(output, 1)
                if label != prediction:
                    print("Prediction for this image is incorrect.")

                # write the labels and pred
                with open(os.path.join(root, 'prediction.txt'), 'w') as pred_log:
                    for k in range(num_classes):
                        pred_log.write('pred: %d, label: %d\n' % (prediction+1, label+1))

                # upsample the assignment and transform the attention correspondingly
                assign_reshaped = torch.nn.functional.interpolate(assign.data.cpu(), size=(height, height), mode='bilinear', align_corners=False)
                # make the background part the first part
                assign_reshaped[:, [0, num_landmarks], :, :] = assign_reshaped[:, [num_landmarks, 0], :, :]

                # generate the one-channel hard assignment via argmax
                assign = torch.argmax(assign_reshaped, dim=1)

                # colorize and save the assignment
                plot_assignment(root, assign.squeeze(0).numpy(), num_landmarks + 1)

                # collect the assignment for the final image array
                color_assignment_name = os.path.join(root, 'assignment.png')
                color_assignment = mpimg.imread(color_assignment_name)
                axarr_assign[j, col_id].imshow(color_assignment)
                axarr_assign[j, col_id].scatter(200, 200, marker='x', c='black')
                axarr_assign[j, col_id].axis('off')

        # save the array version
        os.makedirs('./visualization/collected', exist_ok=True)
        f_assign.savefig(os.path.join('./visualization/collected', args.load+'.png'))

        print('Visualization finished!')

# main method
if __name__ == '__main__':
    main()
