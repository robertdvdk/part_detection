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

# dataset = "WHALE"
# dataset = "PIM"
dataset = "CUB"

def train(net, optimizer, train_loader, device, model_name, epoch,
          epoch_leftoff, loss_fn, all_losses=None):
    # Training
    if all_losses:
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
        anchor, maps, scores, features, classif = net(sample[0].to(device))
        ### FORWARD PASS OF ROTATED IMAGES
        rot_img, rot_angle = rotate_image([90, 45, 60, -45, -60, -90], sample[0])
        flip_img = rot_img
        with torch.no_grad():
            _, equiv_map, _, _, _ = net(flip_img.to(device))

        # Classification loss for landmarks by themselves, and for
        # the attention layer
        loss_class_landmarks = loss_fn(scores[:, :, 0:-1].mean(-1), lab.to(device)).mean()
        loss_class_landmarks = loss_class_landmarks * l_class_lnd
        loss_class_attention = loss_fn(classif, lab.to(device)).mean()
        loss_class_attention = loss_class_attention * l_class_att

        for j in range(scores.shape[0]):
            sorted_classification_lnd, sorted_indices_lnd = scores.mean(-1)[j, :].softmax(0).sort(descending=True)
            sorted_classification, sorted_indices = classif[j, :].softmax(0).sort(descending=True)
            topk_class_lnd.append(list(sorted_indices_lnd).index(lab[j]))
            top_class_lnd.append(sorted_indices_lnd[0])
            topk_class.append(list(sorted_indices).index(lab[j]))
            top_class.append(sorted_indices[0])

        # Get landmark coordinates
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps)

        # Concentration loss
        loss_conc_x = ((loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
        loss_conc_y = ((loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
        loss_conc = (loss_conc_x + loss_conc_y) * maps
        loss_conc = (loss_conc[:, 0:-1, :, :].mean()) * l_conc

        # MAX LOSS PER BATCH INSTEAD OF PER IMAGE
        loss_max = torch.nn.functional.avg_pool2d(maps[:, :, 2:-2, 2:-2], 3, stride=1).max(-1)[0].max(-1)[0].max(0)[0].mean()
        loss_max = (1 - loss_max) * l_max

        ### Orthogonality loss
        normed_feature = torch.nn.functional.normalize(anchor, dim=1)
        similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
        similarity = torch.sub(similarity, torch.eye(net.num_landmarks + 1).to(device))
        orth_loss = torch.mean(torch.square(similarity))
        loss_orth = orth_loss * l_orth

        ## Equivariance loss: calculate rotated landmarks distance
        flip_back = equiv_map
        rot_back = torchvision.transforms.functional.rotate(flip_back, -rot_angle)
        num_elements_per_map = maps.shape[-2] * maps.shape[-1]
        cos_sim_equiv = torch.nn.functional.cosine_similarity(torch.reshape(maps[:, 0:-1, :, :], (-1, net.num_landmarks, num_elements_per_map)), torch.reshape(rot_back[:, 0:-1, :, :].detach(), (-1, net.num_landmarks, num_elements_per_map)), -1)
        del rot_back, equiv_map, flip_back
        loss_equiv = (1 - torch.mean(cos_sim_equiv)) * l_equiv

        total_loss = loss_conc + loss_max + loss_class_attention + loss_orth + loss_equiv + loss_class_landmarks
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
    all_losses = running_loss_conc, running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att
    with open(f'./results_{experiment}/res.txt', 'a') as fopen:
        fopen.write(f'Epoch: {epoch}\n')
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
        anchor, maps, scores, feature_tensor, classification = net(sample[0].to(device))
        scores = scores.detach().cpu()
        all_scores.append(scores)
        lab = sample[1]
        all_labels.append(lab)

        for j in range(scores.shape[0]):
            sorted_classification_lnd, sorted_indices_lnd = scores.mean(-1)[j, :].softmax(0).sort(descending=True)
            sorted_classification, sorted_indices = classification[j, :].softmax(0).sort(descending=True)
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
            show_maps(sample[0], maps, loc_x, loc_y, epoch, experiment, savefig)

    top1acc = str((np.array(topk_class) == 0).mean())
    top5acc = str((np.array(topk_class) < 5).mean())
    top1acclnd = str((np.array(topk_class_lnd) == 0).mean())
    top5acclnd = str((np.array(topk_class_lnd) < 5).mean())
    print(f"Att validation top 1: {top1acc}")
    print(f"Att validation top 5: {top5acc}")
    print(f"Lnd validation top 1: {top1acclnd}")
    print(f"Lnd validation top 5: {top5acclnd}")
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
