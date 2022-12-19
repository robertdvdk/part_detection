"""
Author: Robert van der Klis

Provides some auxiliary functions for the main module
"""


# Import statements
import torch
import torchvision, torchvision.transforms.functional
import numpy as np
import skimage
import matplotlib.pyplot as plt
import os


# Function definitions
def download_whaledataset(dir):
    path = f'{dir}/happyWhale'
    if not os.path.exists(path):
        try:
            os.mkdir(path)
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.competition_download_files('humpback-whale-identification',
                                           path=path)

            import zipfile

            with zipfile.ZipFile(path, 'r') as zipref:
                zipref.extractall(path)
        except Exception as e:
            os.rmdir(path)
            print(e)
            raise RuntimeError("Unable to download Kaggle files! Please read README.md")


def landmark_coordinates(maps, device):
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

def rotate_image(rotations, image):
    angle = int(np.random.choice(rotations))
    rot_img = torchvision.transforms.functional.rotate(image, angle)
    return rot_img, angle

def flip_image(image, threshold):
    flip = np.random.random()
    if flip > threshold:
        flip_img = torchvision.transforms.functional.vflip(image)
    else:
        flip_img = image
    return flip_img, flip > threshold

def landmarks_to_rgb(maps):

    colors = [[0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],

    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],

    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25]]

    rgb = np.zeros((maps.shape[1],maps.shape[2],3))

    for m in range(maps.shape[0]):

        for c in range(3):

            rgb[:,:,c] += maps[m,:,:]*colors[m][c]

    return rgb

def show_maps(ims,maps,loc_x,loc_y, epoch, experiment):
    ''' Plot images, attention maps and landmark centroids.
    Args:
    ims: Torch tensor of images, [batch,3,width_im,height_im]
    maps: Torch tensor of attention maps, [batch, number of maps, width_map, height_map]
    loc_x, loc_y: centroid coordinates, [batch, 0, number of maps]
    '''
    colors = [[0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25],
    [0.75,0,0],[0,0.75,0],[0,0,0.75],[0.5,0.5,0],[0.5,0,0.5],[0,0.5,0.5],[0.75,0.25,0],[0.75,0,0.25],[0,0.75,0.25]]
    fig,axs = plt.subplots(3,3)
    i = 0
    for ax in axs.reshape(-1):
        if i<maps.shape[0]:
            landmarks = landmarks_to_rgb( maps[i,0:-1,:,:].detach().cpu().numpy()) #* feature_magnitudes[i,:,:].unsqueeze(-1).detach().cpu().numpy()
            ax.imshow((skimage.transform.resize(landmarks, (256, 256)) + skimage.transform.resize((ims[i, :, :, :].permute(1, 2, 0).numpy() * 255), (256, 256))))
            ax.scatter(loc_y[i,0:-1].detach().cpu()*256/maps.shape[-1],loc_x[i,0:-1].detach().cpu()*256/maps.shape[-1],c=colors[0:loc_x.shape[1]-1],marker='x')
        i += 1

    plt.savefig(f'./results_{experiment}/{epoch}_{np.random.randint(0, 10)}')

    # plt.show()

def get_epoch(experiment):
    files = os.listdir(f'results_{experiment}')
    epoch = 0
    for f in files:
        if '_' in f:
            fepoch = int(f.split('_')[0])
            if fepoch > epoch:
                epoch = fepoch
    return epoch

def main():
    ### Helper module, no main function
    pass

if __name__ == "__main__":
    main()
