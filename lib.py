"""
Provides some auxiliary functions for the main module
"""


# Import statements
import torch
import numpy as np
import skimage
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as visionF

COLORS = [[0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25]]

# Function definitions
def landmark_coordinates(maps: torch.Tensor, device: torch.device) -> \
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the coordinates of the landmarks from the attention maps
    Parameters
    ----------
    maps: Tensor, [batch_size, number of parts, width_map, height_map]
        Attention maps
    device: torch.device
        The device to use

    Returns
    -------
    loc_x: Tensor, [batch_size, 0, number of parts]
        The centroid x coordinates
    loc_y: Tensor, [batch_size, 0, number of parts]
        The centroid y coordinates
    grid_x: Tensor, [batch_size, 0, width_map]
        The x coordinates of the attention maps
    grid_y: Tensor, [batch_size, 0, height_map]
        The y coordinates of the attention maps
    """
    grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                    torch.arange(maps.shape[3]))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

    map_sums = maps.sum(3).sum(2).detach()
    maps_x = grid_x * maps
    maps_y = grid_y * maps
    loc_x = maps_x.sum(3).sum(2) / map_sums
    loc_y = maps_y.sum(3).sum(2) / map_sums
    return loc_x, loc_y, grid_x, grid_y


def rigid_transform(img: torch.Tensor, angle: int, translate: [int], scale: float, invert: bool=False):
    """
    Affine transforms input image
    Parameters
    ----------
    img: torch.Tensor
        Input image
    angle: int
        Rotation angle between -180 and 180 degrees
    translate: [int]
        Sequence of horizontal/vertical translations
    scale: float
        How to scale the image
    invert: bool
        Whether to invert the transformation

    Returns
    ----------
    img: torch.Tensor
        Transformed image
    """
    shear = 0
    bilinear = visionF.InterpolationMode.BILINEAR
    if not invert:
        img = visionF.affine(img, angle, translate, scale, shear,
                             interpolation=bilinear)
    else:
        translate = [-t for t in translate]
        img = visionF.affine(img, 0, translate, 1, shear)
        img = visionF.affine(img, -angle, [0, 0], 1/scale, shear)
    return img


def landmarks_to_rgb(maps):
    """
    Converts the attention maps to maps of colors
    Parameters
    ----------
    maps: Tensor, [number of parts, width_map, height_map]
        The attention maps to display

    Returns
    ----------
    rgb: Tensor, [width_map, height_map, 3]
        The color maps
    """
    rgb = np.zeros((maps.shape[1],maps.shape[2],3))
    for m in range(maps.shape[0]):
        for c in range(3):
            rgb[:, :, c] += maps[m, :, :] * COLORS[m % 25][c]
    return rgb


def save_maps(X: torch.Tensor, maps: torch.Tensor, epoch: int, model_name: str, device: torch.device) -> None:
    """
    Plot images, attention maps and landmark centroids.
    Parameters
    ----------
    X: Tensor, [batch_size, 3, width_im, height_im]
        Input images on which to show the attention maps
    maps: Tensor, [batch_size, number of parts, width_map, height_map]
        The attention maps to display
    epoch: int
        The current epoch
    model_name: str
        The name of the model
    device: torch.device
        The device to use

    Returns
    -------
    """
    grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]), torch.arange(maps.shape[3]))
    grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)
    map_sums = maps.sum(3).sum(2).detach()
    maps_x = grid_x * maps
    maps_y = grid_y * maps
    loc_x = maps_x.sum(3).sum(2) / map_sums
    loc_y = maps_y.sum(3).sum(2) / map_sums
    fig, axs = plt.subplots(3, 3)
    i = 0
    for ax in axs.reshape(-1):
        if i < maps.shape[0]:
            landmarks = landmarks_to_rgb( maps[i,:-1,:,:].detach().cpu().numpy())
            ax.imshow((skimage.transform.resize(landmarks, (256, 256))
                       + skimage.transform.resize((X[i, :, :, :].permute(1, 2, 0).numpy()), (256, 256))))
            x_coords = loc_y[i,0:-1].detach().cpu()*256/maps.shape[-1]
            y_coords = loc_x[i,0:-1].detach().cpu()*256/maps.shape[-1]
            cols = COLORS[0:loc_x.shape[1]-1]
            n = np.arange(loc_x.shape[1])
            for xi, yi, col_i, mark in zip(x_coords, y_coords, cols, n):
                ax.scatter(xi, yi, color=col_i, marker=f'${mark}$')
        i += 1
    plt.savefig(f'./results_{model_name}/{epoch}_{np.random.randint(0, 10)}')
    plt.close()


def get_epoch(model_name):
    """
    Return the last epoch saved by the model
    Parameters
    ----------
    model_name: string
        The name of the model

    Returns
    ----------
    epoch: int
        The last epoch saved by the model
    """
    files = os.listdir(f'../results_{model_name}')
    epoch = 0
    for f in files:
        if '_' in f:
            fepoch = int(f.split('_')[0])
            if fepoch > epoch:
                epoch = fepoch
    return epoch

if __name__ == "__main__":
    pass
