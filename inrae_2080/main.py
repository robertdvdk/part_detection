from lib import show_maps, landmark_coordinates, rotate_image, flip_image, get_epoch
from datasets import WhaleDataset, PartImageNetDataset, CUBDataset
from nets import Net, LandmarkNet

import os
from typing import Union, Any, Optional, List

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from torch import Tensor
import torch.multiprocessing

from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet
from tqdm import tqdm

import matplotlib.pyplot as plt


# to avoid error "too many files open"
torch.multiprocessing.set_sharing_strategy('file_system')

# Used to name the .pt file and to store results
experiment = "cub_1mhablock_classtoken_dim300_dropout05_lr-3"
if not os.path.exists(f'./results_{experiment}'):
    os.mkdir(f'./results_{experiment}')
# Loss hyperparameters
l_max = 1
# l_max = 0

l_equiv = 1
# l_equiv = 0

l_conc = 1

l_orth = 1
# l_orth = 0

l_class = 1

# l_comp = 1
l_comp = 0

# dataset = "WHALE"
# dataset = "PIM"
dataset = "CUB"

def train(net: torch.nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device, model_name: str,epoch: int, epoch_leftoff, all_losses: list = None):
    # Training
    if all_losses:
        running_loss_conc, running_loss_mean, running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att = all_losses
    elif not all_losses and epoch != 0:
        print('Please pass the losses of the previous epoch to the training function')
    classif_loss = torch.nn.CrossEntropyLoss()
    # TODO reduce learning rate to 10-3
    # TODO train resnet50 w/o landmarks
    # TODO part localization evaluation
    # TODO one histogram of counts per landmark, get entropy of histogram, and get entropy of average of all histograms

    optimizer = torch.optim.Adam(
        [{'params': list(net.parameters())[0:1], 'lr': 1e-3}, # nn.parameter for the class token is on the top of the list
         {'params': list(net.parameters())[1:-4], 'lr': 1e-4},
         {'params': list(net.parameters())[-4:], 'lr': 1e-3}])
    net.train()
    all_training_vectors = []
    all_training_labels = []
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    iter_loader = iter(train_loader)
    topk_class = []
    top_class = []
    for i in range(len(train_loader)):
        sample = next(iter_loader)
        lab = sample[1]

        with torch.no_grad():
            # anchor, _, _, _ = net(sample[0].to(device))
            anchor, _, _, _, _ = net(sample[0].to(device))
            all_training_vectors.append(anchor.cpu())
            all_training_labels.append(sample[1])
        ### DATA AUGMENTATION PROCEDURES
        # Data augmentation 1: Flip the image
        if np.random.rand() > 0.5:
            sample[0] = sample[0].flip(-1)

        # Data augmentation 2: random transform the image
        angle = np.random.randn() * 0.1
        scale = np.random.rand() * 0.2 + 0.9
        sample[0] = torchvision.transforms.functional.affine(sample[0],
                                                             angle=angle * 180 / np.pi,
                                                             translate=[0,0],
                                                             scale=scale,
                                                             shear=0)

        anchor, maps, scores_anchor, feature_tensor, classif_anchor = net(sample[0].to(device))

        ### FORWARD PASS OF ROTATED IMAGES
        rot_img, rot_angle = rotate_image([0, 90, 180, 270], sample[0])
        if rot_angle == 0:
            flip_img, is_flipped = flip_image(rot_img, 1)
        else:
            flip_img, is_flipped = flip_image(rot_img, 0.5)
        # _, equiv_map, _, _ = net(flip_img.to(device))
        _, equiv_map, _, _, _ = net(flip_img.to(device))

        # Classification loss for anchor and positive samples
        loss_class_landmarks = classif_loss(scores_anchor.mean(-1), lab.to(
            device)) / 2
        loss_class_attention = classif_loss(classif_anchor, lab.to(device)) / 2

        for j in range(classif_anchor.shape[0]):
            sorted_classification, sorted_indices = classif_anchor[j, :].softmax(0).sort(descending=True)
            topk_class.append(list(sorted_indices).index(lab[j]))
            top_class.append(sorted_indices[0])

        # loss_class = torch.Tensor([0.]).to(device)
        loss_class = (loss_class_landmarks + loss_class_attention)*l_class

        # Get landmark coordinates
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps, device)

        # Concentration loss
        loss_conc_x = (loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) ** 2
        loss_conc_y = (loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y) ** 2
        loss_conc = ((loss_conc_x + loss_conc_y)) * maps
        loss_conc = (loss_conc[:, 0:-1, :, :].mean()) * l_conc

        # MAX LOSS PER BATCH INSTEAD OF PER IMAGE
        loss_max = maps.max(-1)[0].max(-1)[0].max(0)[0].mean()
        # loss_max = maps.max(-1)[0].max(-1)[0].mean()
        loss_max = (1 - loss_max)*l_max

        ### Orthogonality loss
        normed_feature = torch.nn.functional.normalize(anchor, dim=1)
        similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
        similarity = torch.sub(similarity, torch.eye(net.num_landmarks).to(device))
        orth_loss = torch.mean(torch.square(similarity))
        loss_orth = orth_loss * l_orth
        # loss_orth = torch.Tensor([0.]).to(device)

        ### Compositionality loss
        upsampled_maps = torch.nn.functional.interpolate(maps, size=(256, 256), mode='bilinear')
        # CHANGE TO PER IMAGE INSTEAD OF PER BATCH
        random_landmark = np.random.randint(0, net.num_landmarks)
        random_map = upsampled_maps[:, random_landmark]
        map_argmax = torch.argmax(random_map, axis=0)
        mask = torch.where(map_argmax==random_landmark, 1, 0)
        # Permute dimensions: sample[0] is 12x3x256x256, random_map is 12x256x256
        # permute sample[0] to 3x12x256x256 so we can multiply them
        masked_imgs = torch.permute((torch.permute(sample[0], (1, 0, 2, 3))).to(device) * mask, (1, 0, 2, 3))
        # _, _, _, comp_featuretensor = net(masked_imgs)
        _, _, _, comp_featuretensor, _ = net(masked_imgs)
        masked_feature = (maps[:, random_landmark, :, :].unsqueeze(-1).permute(0,3,1,2) * comp_featuretensor).mean(2).mean(2)
        unmasked_feature = anchor[:, :, random_landmark]
        cos_sim_comp = torch.nn.functional.cosine_similarity(masked_feature.detach(), unmasked_feature, dim=-1)
        comp_loss = 1 - torch.mean(cos_sim_comp)
        loss_comp = comp_loss * l_comp
        # loss_comp = torch.Tensor([0.]).to(device)


        ### Equivariance loss: calculate rotated landmarks distance
        if is_flipped:
            flip_back = torchvision.transforms.functional.hflip(equiv_map)
        else:
            flip_back = equiv_map
        rot_back = torchvision.transforms.functional.rotate(flip_back, 360-rot_angle)
        cos_sim_equiv = torch.nn.functional.cosine_similarity(torch.reshape(maps[:, 0:-1, :, :], (-1, net.num_landmarks, 1024)), torch.reshape(rot_back[:, 0:-1, :, :], (-1, net.num_landmarks, 1024)), -1)
        loss_equiv = (1 - torch.mean(cos_sim_equiv)) * l_equiv
        # loss_equiv = torch.Tensor([0.]).to(device)

        loss_mean = maps[:, 0:-1, :, :].mean()

        total_loss = loss_conc + loss_max + loss_class + loss_equiv + loss_orth + loss_comp
        # total_loss = loss + loss_conc + loss_max + loss_class + loss_equiv + loss_orth + loss_comp
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch == epoch_leftoff and i == 0:
            running_loss_conc = loss_conc.item()
            running_loss_mean = loss_mean.item()
            running_loss_max = loss_max.item()
            running_loss_class_lnd = loss_class_landmarks.item()
            running_loss_equiv = loss_equiv.item()
            running_loss_orth = loss_orth.item()
            running_loss_class_att = loss_class_attention.item()
        else:
            running_loss_conc = 0.99 * running_loss_conc + 0.01 * loss_conc.item()
            running_loss_mean = 0.99 * running_loss_mean + 0.01 * loss_mean.item()
            running_loss_max = 0.99 * running_loss_max + 0.01 * loss_max.item()
            running_loss_class_lnd = 0.99 * running_loss_class_lnd + 0.01 * loss_class_landmarks.item()

            running_loss_equiv = 0.99 * running_loss_equiv + 0.01 * loss_equiv.item()

            running_loss_orth = 0.99 * running_loss_orth + 0.01 * loss_orth.item()

            running_loss_class_att = 0.99 * running_loss_class_att + 0.01 * loss_class_attention.item()

        pbar.set_description(
            "Cnc: %.3f, M: %.3f, Lnd: %.3f, Eq: %.3f, Or: %.3f, Att: %.3f" % (
                running_loss_conc,
                running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att))

        pbar.update()
    pbar.close()
    torch.save(net.cpu().state_dict(), model_name)
    all_losses = running_loss_conc, running_loss_mean, running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att
    with open(f'./results_{experiment}/res.txt', 'a') as fopen:
        fopen.write(f'Epoch: {epoch}\n')
        fopen.write("Cnc: %.3f, M: %.3f, Lnd: %.3f, Eq: %.3f, Or: %.3f, Att: %.3f\n" % (
                running_loss_conc,
                running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att))
        fopen.write(f"Training top 1: {str((np.array(topk_class)==0).mean())}\n"
                    f"Training top 5: {str((np.array(topk_class)<5).mean())}\n")
    return net, all_losses

def validation(device: torch.device, net: torch.nn.Module, val_loader: torch.utils.data.DataLoader, epoch, only_test):
    net.eval()
    net.to(device)
    pbar: tqdm = tqdm(val_loader, position=0, leave=True)
    top_class: list[Any] = []
    names: list[Any] = []
    topk_class: list[int] = []
    diff_to_second: list[float] = []
    topk_lm_class: Optional[List] = None
    class_lm: Optional[List] = None
    all_scores: list[Any] = []
    all_labels: list[Any] = []
    l = 0
    all_maxes = torch.Tensor().to(device)
    for i, sample in enumerate(pbar):
        feat, maps, scores, feature_tensor, classification = net(sample[0].to(device))
        scores = scores.detach().cpu()
        all_scores.append(scores)
        lab = sample[1]
        all_labels.append(lab)

        for j in range(classification.shape[0]):
            sorted_classification, sorted_indices = classification[j, :].softmax(0).sort(descending=True)
            topk_class.append(list(sorted_indices).index(lab[j]))
            top_class.append(sorted_indices[0])

        # Get landmark coordinates
        grid_x: Tensor
        grid_y: Tensor
        grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                        torch.arange(maps.shape[3]))
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

        map_sums = maps.sum(3).sum(2).detach()
        maps_x = grid_x * maps
        maps_y = grid_y * maps
        loc_x = maps_x.sum(3).sum(2) / map_sums
        loc_y = maps_y.sum(3).sum(2) / map_sums

        map_max = maps.max(-1)[0].max(-1)[0][:, :-1].detach()
        all_maxes = torch.cat((all_maxes, map_max), 0)


        if np.random.random() < 0.05:
            if only_test:
                savefig=False
            else:
                savefig=True
            show_maps(sample[0], maps, loc_x, loc_y, epoch, experiment, savefig)

    top1acc = str((np.array(topk_class)==0).mean())
    top5acc = str((np.array(topk_class)<5).mean())
    print(top1acc)
    print(top5acc)
    colors = [[0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],[0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25],[0, 0.75, 0.25],
              [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],[0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25],[0, 0.75, 0.25],
              [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],[0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25],[0, 0.75, 0.25]]
    show_plots = False
    if show_plots:
        fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
        for i in range(10):
            axs[i//5, i%5].hist(all_maxes[:, i].cpu().numpy(), range=(0, 1), bins=25, color=colors[i])
        plt.show()
    if not only_test:
        print(top1acc)
        print(top5acc)
        with open(f'results_{experiment}/res.txt', 'a') as fopen:
            fopen.write(f"Validation top 1: {top1acc} \n")
            fopen.write(f"Validation top 5: {top5acc} \n")
    pbar.close()

def main():
    print(torch.cuda.is_available())

    whale_path: str = "./datasets/happyWhale"
    pim_path: str = "./datasets/pim"
    cub_path: str = "./datasets/cub/CUB_200_2011"
    np.random.seed(1)
    if dataset == "WHALE":
        dataset_train: WhaleDataset = WhaleDataset(whale_path, mode='train')
        dataset_val: WhaleDataset = WhaleDataset(whale_path, mode='val')
        dataset_full: WhaleDataset = WhaleDataset(whale_path, mode='no_set', minimum_images=0,
                                    alt_data_path='Teds_OSM')

    elif dataset == "PIM":
        dataset_train: WhaleDataset = PartImageNetDataset(pim_path,mode='train')
        dataset_val: WhaleDataset = PartImageNetDataset(pim_path,mode='val')


    elif dataset == "CUB":
        dataset_train = CUBDataset(cub_path, mode='train')
        dataset_val = CUBDataset(cub_path, mode='val', train_samples=dataset_train.trainsamples)

        # don't use for now
        dataset_test = CUBDataset(cub_path, mode='test')


    batch_size = 24
    train_loader: DataLoader[Any] = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    val_loader: DataLoader[Any] = torch.utils.data.DataLoader(dataset=dataset_val,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    number_epochs: int = 50
    model_name: str = f'{experiment}.pt'
    model_name_init: str = f'{experiment}.pt'
    warm_start: bool = False
    do_only_test: bool = False

    num_landmarks: int = 10

    basenet: ResNet = torchvision.models.resnet18(pretrained=True)
    net: Union[Net,LandmarkNet]

    if dataset=="WHALE":
        num_cls = 2000
    elif dataset=="CUB":
        num_cls = 200
    elif dataset=="PIM":
        num_cls = 160

    net = LandmarkNet(basenet, num_landmarks, num_classes=num_cls)

    if warm_start:
        net.load_state_dict(torch.load(model_name_init), strict=False)
        epoch_leftoff = get_epoch(experiment) + 1
    else:
        epoch_leftoff = 0
    net.to(device)

    if do_only_test:
        epoch_leftoff = 0
        number_epochs = 1

    all_losses = []
    for epoch in range(epoch_leftoff, number_epochs):
        if not do_only_test:
            if all_losses:
                net, all_losses = train(net, train_loader, device, model_name, epoch, 0, all_losses)
            else:
                net, all_losses = train(net, train_loader, device, model_name, epoch, epoch_leftoff)
            print(f'Validation accuracy in epoch {epoch}:')
            validation(device, net, val_loader, epoch, do_only_test)
        # Validation
        else:
            print('Validation accuracy with saved network:')
            validation(device, net, val_loader, epoch, do_only_test)

if __name__=="__main__":
    main()
