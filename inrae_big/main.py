from lib import show_maps, landmark_coordinates, rotate_image, flip_image, \
    get_epoch, grad_reverse
from datasets import WhaleDataset, PartImageNetDataset, CUBDataset, CUB200
from nets import Net, LandmarkNet, NewLandmarkNet

import os
from typing import Union, Any, Optional, List

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from torch import Tensor
import torch.multiprocessing
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torchvision.models import resnet101, ResNet101_Weights
from tqdm import tqdm

import matplotlib.pyplot as plt

# to avoid error "too many files open"
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

# Used to name the .pt file and to store results
experiment = "8parts_fulldataset_1mha_adam30_sgd20_sgdscratchx1"
if not os.path.exists(f'./results_{experiment}'):
    os.mkdir(f'./results_{experiment}')
# Loss hyperparameters
l_max = 1
# l_max = 0

l_equiv = 1
# l_equiv = 0

l_conc = 1000
# l_conc = 0

l_orth = 1
# l_orth = 0

l_class_lnd = 1

l_class_att = 1

# l_comp = 1
l_comp = 0

# dataset = "WHALE"
# dataset = "PIM"
dataset = "CUB"

BASENET = False


def train(net, optimizer, train_loader, device, model_name, epoch,
          epoch_leftoff, loss_fn, all_losses=None):
    # Training
    if all_losses:
        if BASENET:
            running_loss_class = all_losses
        else:
            running_loss_conc, running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att = all_losses
    elif not all_losses and epoch != 0:
        print(
            'Please pass the losses of the previous epoch to the training function')
    net.train()
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    iter_loader = iter(train_loader)
    topk_class = []
    top_class = []
    topk_class_lnd = []
    top_class_lnd = []
    for i in range(len(train_loader)):
        sample = next(iter_loader)
        lab = sample[1]
        if BASENET:
            _, _, scores, _, _ = net(sample[0].to(device))
            loss_class = loss_fn(scores, lab.to(device)).mean()
            for j in range(scores.shape[0]):
                sorted_classification, sorted_indices = scores[j, :].softmax(
                    0).sort(descending=True)
                topk_class.append(list(sorted_indices).index(lab[j]))
                top_class.append(sorted_indices[0])
            loss_class.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch == epoch_leftoff and i == 0:
                running_loss_class = loss_class.item()
            else:
                running_loss_class = 0.99 * running_loss_class + 0.01 * loss_class.item()
                pbar.set_description(f"Cls: {round(running_loss_class, 3)}")
        else:
            anchor, maps, scores_anchor, feature_tensor, classif_anchor = net(
                sample[0].to(device))
            ### FORWARD PASS OF ROTATED IMAGES
            rot_img, rot_angle = rotate_image([90, 45, 60, -45, -60, -90], sample[0])
            flip_img = rot_img
            is_flipped = False
            # rot_img, rot_angle = sample[0], 0
            #if rot_angle == 0:
            #    flip_img, is_flipped = flip_image(rot_img, 0.5)
            #else:
            #    flip_img, is_flipped = flip_image(rot_img, 0.5)

            with torch.no_grad():
                _, equiv_map, _, _, _ = net(flip_img.to(device))

            # Classification loss for anchor and positive samples
            loss_class_landmarks = loss_fn(scores_anchor[:, :, 0:-1].mean(-1),
                                           lab.to(device)).mean() * 1
            # loss_class_landmarks = loss_fn(scores_anchor[:,:,0:-1], lab.unsqueeze(-1).repeat(1,scores_anchor[:,:,0:-1].shape[-1]).to(device)) * 1
            # loss_class_landmarks = (loss_class_landmarks * (maps*2-1).relu().max(-1)[0].max(-1)[0][:,0:-1].detach()).mean()

            # loss_class_landmarks_bg = loss_fn(grad_reverse(scores_anchor)[:,:,-1:], lab.unsqueeze(-1).repeat(1,scores_anchor[:,:,-1:].shape[-1]).to(device)) * 1
            # loss_class_landmarks_bg = (loss_class_landmarks_bg * maps.max(-1)[0].max(-1)[0][:,-1:].detach()).mean()
            # loss_class_attention = torch.Tensor([0.]).to(device)
            loss_class_attention = loss_fn(classif_anchor,
                                           lab.to(device)).mean() * 1

            for j in range(scores_anchor.shape[0]):
                sorted_classification_lnd, sorted_indices_lnd = scores_anchor.mean(-1)[j, :].softmax(0).sort(descending=True)
                sorted_classification, sorted_indices = classif_anchor[j,
                                                        :].softmax(0).sort(
                    descending=True)
                topk_class_lnd.append(list(sorted_indices_lnd).index(lab[j]))
                top_class_lnd.append(sorted_indices_lnd[0])
                topk_class.append(list(sorted_indices).index(lab[j]))
                top_class.append(sorted_indices[0])

            # loss_class = torch.Tensor([0.]).to(device)

            # Get landmark coordinates
            loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps)

            # Concentration loss
            loss_conc_x = ((loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) /
                           grid_x.shape[-1]) ** 2
            loss_conc_y = ((loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y) /
                           grid_y.shape[-2]) ** 2
            # loss_conc_x = (loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) ** 2
            # loss_conc_y = (loc_x.unsqueeze(-1).unsqueeze(-1) - grid_y) ** 2
            # loss_conc = (loss_conc_x + loss_conc_y).tanh() * maps
            loss_conc = (loss_conc_x + loss_conc_y) * maps
            loss_conc = (loss_conc[:, 0:-1, :, :].mean()) * l_conc
            # loss_conc = torch.Tensor([0.]).to(device)
            #
            # MAX LOSS PER BATCH INSTEAD OF PER IMAGE
            loss_max = \
            torch.nn.functional.avg_pool2d(maps[:, :, 2:-2, 2:-2], 3,
                                           stride=1).max(-1)[0].max(-1)[0].max(
                0)[0].mean()
            # loss_max = maps.max(-1)[0].max(-1)[0].mean()
            loss_max = (1 - loss_max) * l_max
            # loss_max = torch.Tensor([0.]).to(device)
            #
            ### Orthogonality loss
            normed_feature = torch.nn.functional.normalize(anchor, dim=1)
            similarity = torch.matmul(normed_feature.permute(0, 2, 1),
                                      normed_feature)
            similarity = torch.sub(similarity,
                                   torch.eye(net.num_landmarks + 1).to(device))
            orth_loss = torch.mean(torch.square(similarity))

            # orth_loss = -torch.matmul((-anchor.permute(0, 2, 1).detach()).exp(), -anchor).mean()
            loss_orth = orth_loss * l_orth
            # loss_orth = torch.Tensor([0.]).to(device)

            ## Compositionality loss
            # TODO reshuffle pixels to background instead of black background

            # # CHANGE TO PER IMAGE INSTEAD OF PER BATCH
            # random_landmark = np.random.randint(0, net.num_landmarks)
            # random_map = maps[:, random_landmark:random_landmark+1,:,:]
            # random_map_upsampled = torch.nn.functional.interpolate(random_map, size=(sample[0].shape[-2], sample[0].shape[-1]), mode='bilinear')
            # map_argmax = torch.argmax(random_map, axis=0)
            # mask = torch.where(map_argmax==random_landmark, 1, 0)
            # # Permute dimensions: sample[0] is 12x3x256x256, random_map is 12x256x256
            # # permute sample[0] to 3x12x256x256 so we can multiply them
            # masked_imgs = torch.permute((torch.permute(sample[0], (1, 0, 2, 3))).to(device) * mask, (1, 0, 2, 3))

            # with torch.no_grad():
            #    masked_imgs = (sample[0].to(device)*random_map_upsampled)
            #    _, _, _, comp_featuretensor, _ = net(masked_imgs)
            #    masked_feature = (comp_featuretensor*random_map).mean(-1).mean(-1)

            # # _, _, _, comp_featuretensor = net(masked_imgs)
            # _, _, _, comp_featuretensor, _ = net(masked_imgs)
            # masked_feature = (maps[:, random_landmark, :, :].unsqueeze(-1).permute(0,3,1,2) * comp_featuretensor).mean(-1).mean(-1)

            # unmasked_feature = anchor[:, :, random_landmark]
            # cos_sim_comp = torch.nn.functional.cosine_similarity(masked_feature.detach(), unmasked_feature, dim=-1)
            # comp_loss = 1 - torch.mean(cos_sim_comp)
            # loss_comp = comp_loss * l_comp

            loss_comp = torch.Tensor([0.]).to(device)

            ## Equivariance loss: calculate rotated landmarks distance
            if is_flipped:
                flip_back = torchvision.transforms.functional.hflip(equiv_map)
            else:
                flip_back = equiv_map
            rot_back = torchvision.transforms.functional.rotate(flip_back,
                                                                -rot_angle)

            # rotloc_x, rotloc_y, _, _ = landmark_coordinates(rot_back, device)
            # locxy = torch.stack((loc_x, loc_y), dim=1)/grid_x.shape[-1]
            # rotlocxy = torch.stack((rotloc_x, rotloc_y), dim=1)/grid_x.shape[-1]
            # sum over batches, x and y coordinate, then take mean of all landmarks
            # loss_equiv = (locxy - rotlocxy).pow(2).sum(0).sum(0)[0:-1].mean()

            num_elements_per_map = maps.shape[-2] * maps.shape[-1]
            cos_sim_equiv = torch.nn.functional.cosine_similarity(
                torch.reshape(maps[:, 0:-1, :, :],
                              (-1, net.num_landmarks, num_elements_per_map)),
                torch.reshape(rot_back[:, 0:-1, :, :].detach(),
                              (-1, net.num_landmarks, num_elements_per_map)),
                -1)
            del rot_back, equiv_map, flip_back
            # cos_sim_equiv = torch.nn.functional.cosine_similarity(torch.reshape(maps[:, 0:-1, :, :],(-1, net.num_landmarks, 1024)),torch.reshape(rot_back[:, 0:-1, :, :],(-1, net.num_landmarks, 1024)), -1)
            loss_equiv = (1 - torch.mean(cos_sim_equiv)) * l_equiv
            # loss_equiv = torch.Tensor([0.]).to(device)

            total_loss = loss_conc + loss_max + loss_class + loss_orth + loss_equiv + loss_comp
            # total_loss = loss_class_landmarks
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            if epoch == epoch_leftoff and i == 0:
                running_loss_conc = loss_conc.item()
                running_loss_max = loss_max.item()
                running_loss_class_lnd = loss_class_landmarks.item()
                running_loss_equiv = loss_equiv.item()
                running_loss_orth = loss_orth.item()
                running_loss_class_att = loss_class_attention.item()
            else:
                running_loss_conc = 0.99 * running_loss_conc + 0.01 * loss_conc.item()
                running_loss_max = 0.99 * running_loss_max + 0.01 * loss_max.item()
                running_loss_class_lnd = 0.99 * running_loss_class_lnd + 0.01 * loss_class_landmarks.item()
                running_loss_equiv = 0.99 * running_loss_equiv + 0.01 * loss_equiv.item()
                running_loss_orth = 0.99 * running_loss_orth + 0.01 * loss_orth.item()
                running_loss_class_att = 0.99 * running_loss_class_att + 0.01 * loss_class_attention.item()
                pbar.set_description(
                    "Cnc: %.3f, M: %.3f, Lnd: %.3f, Eq: %.3f, Or: %.3f, Att: %.3f" % (
                        running_loss_conc,
                        running_loss_max, running_loss_class_lnd,
                        running_loss_equiv, running_loss_orth,
                        running_loss_class_att))
        pbar.update()
    pbar.close()
    torch.save(net.cpu().state_dict(), model_name)
    if BASENET:
        all_losses = running_loss_class
    else:
        all_losses = running_loss_conc, running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att
    with open(f'./results_{experiment}/res.txt', 'a') as fopen:
        fopen.write(f'Epoch: {epoch}\n')
        if BASENET:
            fopen.write(f"Cls: {round(running_loss_class, 3)}\n")
        else:
            fopen.write(
                "Cnc: %.3f, M: %.3f, Lnd: %.3f, Eq: %.3f, Or: %.3f, Att: %.3f\n" % (
                    running_loss_conc,
                    running_loss_max, running_loss_class_lnd,
                    running_loss_equiv, running_loss_orth,
                    running_loss_class_att))
        fopen.write(
            f"Att training top 1: {str((np.array(topk_class) == 0).mean())}\n"
            f"Att training top 5: {str((np.array(topk_class) < 5).mean())}\n")
        fopen.write(
            f"Lnd training top 1: {str((np.array(topk_class_lnd) == 0).mean())}\n"
            f"Lnd training top 1: {str((np.array(topk_class_lnd) < 5).mean())}\n")
    return net, all_losses


def validation(device: torch.device, net: torch.nn.Module,
               val_loader: torch.utils.data.DataLoader, epoch, only_test):
    net.eval()
    net.to(device)
    pbar: tqdm = tqdm(val_loader, position=0, leave=True)
    top_class = []
    topk_class = []
    top_class_lnd = []
    topk_class_lnd = []
    all_scores = []
    all_labels = []
    all_maxes = torch.Tensor().to(device)
    for i, sample in enumerate(pbar):
        if BASENET:
            _, _, scores, _, _ = net(sample[0].to(device))
            # scores = net(sample[0].to(device))
            scores = scores.detach().cpu()
            lab = sample[1]
            for j in range(scores.shape[0]):
                sorted_classification, sorted_indices = scores[j, :].softmax(
                    0).sort(descending=True)

                topk_class.append(list(sorted_indices).index(lab[j]))
                top_class.append(sorted_indices[0])
        else:
            anchor, maps, scores, feature_tensor, classification = net(
                sample[0].to(device))
            scores = scores.detach().cpu()
            all_scores.append(scores)
            lab = sample[1]
            all_labels.append(lab)

            for j in range(scores.shape[0]):
                sorted_classification_lnd, sorted_indices_lnd = scores.mean(-1)[j,
                                                        :].softmax(0).sort(
                    descending=True)
                sorted_classification, sorted_indices = classification[j,
                                                        :].softmax(0).sort(
                    descending=True)
                topk_class_lnd.append(list(sorted_indices_lnd).index(lab[j]))
                top_class_lnd.append(sorted_indices_lnd[0])
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
                    savefig = False
                else:
                    savefig = True
                show_maps(sample[0], maps, loc_x, loc_y, epoch, experiment,
                          savefig)

    top1acc = str((np.array(topk_class) == 0).mean())
    top5acc = str((np.array(topk_class) < 5).mean())
    top1acclnd = str((np.array(topk_class_lnd) == 0).mean())
    top5acclnd = str((np.array(topk_class_lnd) < 5).mean())
    print(f"Att validation top 1: {top1acc}")
    print(f"Att validation top 5: {top5acc}")
    print(f"Lnd validation top 1: {top1acclnd}")
    print(f"Lnd validation top 5: {top5acclnd}")
    if not BASENET:
        colors = [[0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],
                  [0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0],
                  [0.75, 0, 0.25], [0, 0.75, 0.25],
                  [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],
                  [0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0],
                  [0.75, 0, 0.25], [0, 0.75, 0.25],
                  [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],
                  [0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0],
                  [0.75, 0, 0.25], [0, 0.75, 0.25]]
        fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
        for i in range(len(loc_x) - 1):
            axs[i // 5, i % 5].hist(all_maxes[:, i].cpu().numpy(),
                                    range=(0, 1), bins=25, color=colors[i])
        plt.show()
    if not only_test:
        with open(f'results_{experiment}/res.txt', 'a') as fopen:
            fopen.write(f"Att validation top 1: {top1acc} \n")
            fopen.write(f"Att validation top 5: {top5acc} \n")
            fopen.write(f"Lnd validation top 1: {top1acclnd} \n")
            fopen.write(f"Lnd validation top 5: {top5acclnd} \n")
    pbar.close()


# def ablation():
#     print(torch.cuda.is_available())
#
#     cub_path = "./datasets/cub/CUB_200_2011"
#     np.random.seed(1)
#     height = 448
#     train_transforms = transforms.Compose([
#         transforms.Resize(size=height),
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(0.1),
#         torchvision.transforms.RandomRotation(45),
#         torchvision.transforms.RandomResizedCrop(height, scale=(0.6, 1.0),
#                                                  ratio=(0.9, 1.1)),
#         transforms.ToTensor(),
#     ])
#     test_transforms = transforms.Compose([
#         transforms.Resize(size=height),
#         transforms.CenterCrop(size=height),
#         transforms.ToTensor(),
#     ])
#     dataset_train = CUBDataset(cub_path, mode='train',
#                                height=height, transform=train_transforms)
#
#     dataset_test = CUBDataset(cub_path, mode='test',
#                               transform=test_transforms)
#
#     train_batch = 20
#     train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
#                                                batch_size=train_batch,
#                                                shuffle=True, num_workers=4)
#
#     test_batch = 8
#     val_loader = torch.utils.data.DataLoader(dataset=dataset_test,
#                                              batch_size=test_batch,
#                                              shuffle=False, num_workers=4)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     number_epochs = 20
#     model_name = f'{experiment}.pt'
#
#     num_landmarks = 8
#
#     weights = ResNet101_Weights.DEFAULT
#     basenet = resnet101(weights=weights)
#
#     num_cls = 200
#
#     net = NewLandmarkNet(basenet, num_landmarks, num_classes=num_cls,
#                          height=height)
#
#     epoch_leftoff = 0
#     net.to(device)
#
#     all_losses = []
#
#     scratch_layers = ["fc_class_landmarks", "fc_class_attention"]
#     finer_layers = ["fc_landmarks", "fc", "fc_class_final", "mha"]
#     finetune_parameters = []
#     scratch_parameters = []
#     finer_parameters = []
#     for name, p in net.named_parameters():
#         layer_name = name.split('.')[0]
#         if layer_name in scratch_layers:
#             scratch_parameters.append(p)
#         elif layer_name in finer_layers:
#             finer_parameters.append(p)
#         else:
#             finetune_parameters.append(p)
#
#     loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
#     baselr = 1e-4
#     # STEPLR
#     optimizer = torch.optim.Adam(
#         [{'params': scratch_parameters, 'lr': baselr * 10},
#          {'params': finer_parameters, 'lr': baselr * 1},
#          {'params': finetune_parameters, 'lr': baselr},
#          ])
#     # TODO change back to 5?
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.5)
#     for epoch in range(epoch_leftoff, number_epochs):
#         if epoch == 15:
#             optimizer = torch.optim.SGD(
#                 [{'params': scratch_parameters, 'lr': baselr * 1},
#                  {'params': finer_parameters, 'lr': baselr * 1},
#                  {'params': finetune_parameters, 'lr': baselr}],
#                 weight_decay=5e-4, momentum=0.9)
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.5)
#
#         if all_losses:
#             net, all_losses = train(net, optimizer, train_loader, device,
#                                     model_name, epoch, 0, loss_fn,
#                                     all_losses)
#         else:
#             net, all_losses = train(net, optimizer, train_loader, device,
#                                     model_name, epoch, epoch_leftoff,
#                                     loss_fn)
#         scheduler.step()
#         print(f'Validation accuracy in epoch {epoch}:')
#         validation(device, net, val_loader, epoch, False)
#         torch.cuda.empty_cache()


def main():
    print(torch.cuda.is_available())

    whale_path = "./datasets/happyWhale"
    pim_path = "./datasets/pim"
    cub_path = "./datasets/cub/CUB_200_2011"
    # cub_path = "./datasets/cub"
    np.random.seed(1)
    height = 448
    if dataset == "WHALE":
        dataset_train = WhaleDataset(whale_path, mode='train')
        dataset_val = WhaleDataset(whale_path, mode='val')
        dataset_full = WhaleDataset(whale_path, mode='no_set',
                                                  minimum_images=0,
                                                  alt_data_path='Teds_OSM')

    elif dataset == "PIM":
        dataset_train = PartImageNetDataset(pim_path,
                                                          mode='train')
        dataset_val = PartImageNetDataset(pim_path, mode='val')


    elif dataset == "CUB":
        train_transforms = transforms.Compose([
            transforms.Resize(size=height),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1),
            torchvision.transforms.RandomRotation(45),
            torchvision.transforms.RandomResizedCrop(height, scale=(0.6, 1.0),
                                                     ratio=(0.9, 1.1)),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(size=height),
            transforms.CenterCrop(size=height),
            transforms.ToTensor(),
        ])
        # dataset_train = CUBDataset(cub_path, mode='train',
        #                            height=height, transform=train_transforms)
        dataset_train = CUBDataset(cub_path, split=0.9, mode='train',
                                   height=height, transform=train_transforms)
        dataset_val = CUBDataset(cub_path, split=0.9, mode='val',
                                 train_samples=dataset_train.trainsamples,
                                 height=height, transform=test_transforms)
        # dataset_train = CUB200(cub_path, train=True, transform=train_transforms)
        # dataset_val = CUB200(cub_path, train=False, transform=test_transforms)

        # don't use for now
        # dataset_test = CUBDataset(cub_path, mode='test', transform=test_transforms)

    train_batch = 20
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=train_batch,
                                               shuffle=True, num_workers=4)

    test_batch = 8
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val,
                                             batch_size=test_batch,
                                             shuffle=False, num_workers=4)
    # val_loader = torch.utils.data.DataLoader(dataset=dataset_test,
    #                                          batch_size=test_batch,
    #                                          shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=test_batch, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    number_epochs = 20
    model_name = f'{experiment}.pt'
    model_name_init = f'{experiment}.pt'
    warm_start = False
    do_only_test = False

    num_landmarks = 8

    weights = ResNet101_Weights.DEFAULT
    basenet = resnet101(weights=weights)

    if dataset == "WHALE":
        num_cls = 2000
    elif dataset == "CUB":
        num_cls = 200
    elif dataset == "PIM":
        num_cls = 160

    # net = Net(basenet, num_classes=num_cls)
    net = NewLandmarkNet(basenet, num_landmarks, num_classes=num_cls,
                         height=height)

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

    finetune_layers = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    scratch_layers = ["fc_class_landmarks", "fc_class_attention"]
    finer_layers = ["fc_landmarks", "fc", "fc_class_final", "mha"]
    finetune_parameters = []
    scratch_parameters = []
    finer_parameters = []
    for name, p in net.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in scratch_layers:
            scratch_parameters.append(p)
        elif layer_name in finer_layers:
            finer_parameters.append(p)
        else:
            finetune_parameters.append(p)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    baselr = 1e-4
    # STEPLR
    # change lr every 4 epochs
    # run for 15 epochs
    # run 4 epochs with sgd
    optimizer = torch.optim.Adam(
        [{'params': scratch_parameters, 'lr': baselr * 10},
         {'params': finer_parameters, 'lr': baselr * 1},
         {'params': finetune_parameters, 'lr': baselr},
         ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    for epoch in range(epoch_leftoff, number_epochs):
        if epoch == 30:
            optimizer = torch.optim.SGD(
                [{'params': scratch_parameters, 'lr': baselr * 1},
                 {'params': finer_parameters, 'lr': baselr * 1},
                 {'params': finetune_parameters, 'lr': baselr}],
                weight_decay=5e-4, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

        if not do_only_test:
            if all_losses:
                net, all_losses = train(net, optimizer, train_loader, device,
                                        model_name, epoch, 0, loss_fn,
                                        all_losses)
            else:
                net, all_losses = train(net, optimizer, train_loader, device,
                                        model_name, epoch, epoch_leftoff,
                                        loss_fn)
            scheduler.step()
            print(f'Validation accuracy in epoch {epoch}:')
            validation(device, net, val_loader, epoch, do_only_test)
            torch.cuda.empty_cache()
        # Validation
        else:
            print('Validation accuracy with saved network:')
            validation(device, net, val_loader, epoch, do_only_test)


if __name__ == "__main__":
    main()
