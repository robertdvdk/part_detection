from lib import show_maps, landmark_coordinates, rotate_image, flip_image, get_epoch
from datasets import WhaleDataset, PartImageNetDataset, CUBDataset
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

# Used to name the .pt file and to store results
experiment = "cub_resnet101_hires_adamthensgd_nolandmarkloss"
if not os.path.exists(f'./results_{experiment}'):
    os.mkdir(f'./results_{experiment}')
# Loss hyperparameters
l_max = 1
# l_max = 0

l_equiv = 1
# l_equiv = 0

l_conc = 0.25
# l_conc = 0

l_orth = 1
# l_orth = 0

l_class = 1

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
        print('Please pass the losses of the previous epoch to the training function')
    net.train()
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    iter_loader = iter(train_loader)
    topk_class = []
    top_class = []
    for i in range(len(train_loader)):
        sample = next(iter_loader)
        lab = sample[1]
        if BASENET:
            _, _, scores, _, _ = net(sample[0].to(device))
            loss_class = loss_fn(scores, lab.to(device))
            for j in range(scores.shape[0]):
                sorted_classification, sorted_indices = scores[j, :].softmax(0).sort(descending=True)
                topk_class.append(list(sorted_indices).index(lab[j]))
                top_class.append(sorted_indices[0])
            loss_class.backward()
            optimizer.step()
            optimizer.zero_grad()
            if epoch == epoch_leftoff and i == 0:
                running_loss_class = loss_class.item()
            else:
                running_loss_class = 0.99*running_loss_class + 0.01*loss_class.item()
                pbar.set_description(f"Cls: {round(running_loss_class, 3)}")
        else:

            anchor, maps, scores_anchor, feature_tensor, classif_anchor = net(sample[0].to(device))
            ### FORWARD PASS OF ROTATED IMAGES
            rot_img, rot_angle = rotate_image([0, 90, 180, 270], sample[0])
            if rot_angle == 0:
                flip_img, is_flipped = flip_image(rot_img, 1)
            else:
                flip_img, is_flipped = flip_image(rot_img, 0.5)

            with torch.no_grad():
                _, equiv_map, _, _, _ = net(flip_img.to(device))


            # Classification loss for anchor and positive samples
            loss_class_landmarks = loss_fn(scores_anchor.mean(-1), lab.to(device)) * 0
            # loss_class_attention = torch.Tensor([0.]).to(device)
            loss_class_attention = loss_fn(classif_anchor, lab.to(device)) * 1

            for j in range(scores_anchor.shape[0]):
                sorted_classification, sorted_indices = classif_anchor[j, :].softmax(0).sort(descending=True)
                topk_class.append(list(sorted_indices).index(lab[j]))
                top_class.append(sorted_indices[0])

            # loss_class = torch.Tensor([0.]).to(device)
            loss_class = (loss_class_landmarks + loss_class_attention)*l_class

            # Get landmark coordinates
            loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps, device)

            # Concentration loss
            # TODO scale concentration loss to images instead of pixels
            loss_conc_x = (loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) ** 2
            loss_conc_y = (loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y) ** 2
            # loss_conc = (loss_conc_x + loss_conc_y).tanh() * maps
            loss_conc = (loss_conc_x + loss_conc_y) * maps
            loss_conc = (loss_conc[:, 0:-1, :, :].mean()) * l_conc
            # loss_conc = torch.Tensor([0.]).to(device)
            #
            # MAX LOSS PER BATCH INSTEAD OF PER IMAGE
            loss_max = maps.max(-1)[0].max(-1)[0].max(0)[0].mean()
            # loss_max = maps.max(-1)[0].max(-1)[0].mean()
            loss_max = (1 - loss_max)*l_max
            # loss_max = torch.Tensor([0.]).to(device)
            #
            ### Orthogonality loss
            normed_feature = torch.nn.functional.normalize(anchor, dim=1)
            similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
            similarity = torch.sub(similarity, torch.eye(net.num_landmarks).to(device))
            orth_loss = torch.mean(torch.square(similarity))
            loss_orth = orth_loss * l_orth
            # loss_orth = torch.Tensor([0.]).to(device)

            ## Compositionality loss
            # TODO reshuffle pixels to background instead of black background
            # upsampled_maps = torch.nn.functional.interpolate(maps, size=(256, 256), mode='bilinear')
            # # CHANGE TO PER IMAGE INSTEAD OF PER BATCH
            # random_landmark = np.random.randint(0, net.num_landmarks)
            # random_map = upsampled_maps[:, random_landmark]
            # map_argmax = torch.argmax(random_map, axis=0)
            # mask = torch.where(map_argmax==random_landmark, 1, 0)
            # # Permute dimensions: sample[0] is 12x3x256x256, random_map is 12x256x256
            # # permute sample[0] to 3x12x256x256 so we can multiply them
            # masked_imgs = torch.permute((torch.permute(sample[0], (1, 0, 2, 3))).to(device) * mask, (1, 0, 2, 3))
            # # _, _, _, comp_featuretensor = net(masked_imgs)
            # _, _, _, comp_featuretensor, _ = net(masked_imgs)
            # masked_feature = (maps[:, random_landmark, :, :].unsqueeze(-1).permute(0,3,1,2) * comp_featuretensor).mean(2).mean(2)
            # unmasked_feature = anchor[:, :, random_landmark]
            # cos_sim_comp = torch.nn.functional.cosine_similarity(masked_feature.detach(), unmasked_feature, dim=-1)
            # comp_loss = 1 - torch.mean(cos_sim_comp)
            # loss_comp = comp_loss * l_comp
            # loss_comp = torch.Tensor([0.]).to(device)


            ## Equivariance loss: calculate rotated landmarks distance
            # TODO points instead of attention maps
            if is_flipped:
                flip_back = torchvision.transforms.functional.hflip(equiv_map)
            else:
                flip_back = equiv_map
            rot_back = torchvision.transforms.functional.rotate(flip_back, 360-rot_angle)
            cos_sim_equiv = torch.nn.functional.cosine_similarity(torch.reshape(maps[:, 0:-1, :, :], (-1, net.num_landmarks, 3136)), torch.reshape(rot_back[:, 0:-1, :, :], (-1, net.num_landmarks, 3136)), -1)
            # cos_sim_equiv = torch.nn.functional.cosine_similarity(torch.reshape(maps[:, 0:-1, :, :],(-1, net.num_landmarks, 1024)),torch.reshape(rot_back[:, 0:-1, :, :],(-1, net.num_landmarks, 1024)), -1)
            loss_equiv = (1 - torch.mean(cos_sim_equiv)) * l_equiv
            # loss_equiv = torch.Tensor([0.]).to(device)

            total_loss = loss_conc + loss_max + loss_class + loss_equiv + loss_orth
            # total_loss = loss_class_landmarks
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
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
                        running_loss_max, running_loss_class_lnd, running_loss_equiv, running_loss_orth, running_loss_class_att))
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
        if BASENET:
            _, _, scores, _, _ = net(sample[0].to(device))
            # scores = net(sample[0].to(device))
            scores = scores.detach().cpu()
            lab = sample[1]
            for j in range(scores.shape[0]):
                sorted_classification, sorted_indices = scores[j, :].softmax(0).sort(descending=True)
                topk_class.append(list(sorted_indices).index(lab[j]))
                top_class.append(sorted_indices[0])
        else:
            anchor, maps, scores, feature_tensor, classification = net(sample[0].to(device))
            scores = scores.detach().cpu()
            all_scores.append(scores)
            lab = sample[1]
            all_labels.append(lab)

            for j in range(scores.shape[0]):
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
    if not BASENET:
        colors = [[0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],[0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25],[0, 0.75, 0.25],
                  [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],[0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25],[0, 0.75, 0.25],
                  [0.75, 0, 0], [0, 0.75, 0], [0, 0, 0.75], [0.5, 0.5, 0],[0.5, 0, 0.5], [0, 0.5, 0.5], [0.75, 0.25, 0], [0.75, 0, 0.25],[0, 0.75, 0.25]]
        fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
        for i in range(10):
            axs[i//5, i%5].hist(all_maxes[:, i].cpu().numpy(), range=(0, 1), bins=25, color=colors[i])
        plt.show()
    if not only_test:
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
    height=448
    if dataset == "WHALE":
        dataset_train: WhaleDataset = WhaleDataset(whale_path, mode='train')
        dataset_val: WhaleDataset = WhaleDataset(whale_path, mode='val')
        dataset_full: WhaleDataset = WhaleDataset(whale_path, mode='no_set', minimum_images=0,
                                    alt_data_path='Teds_OSM')

    elif dataset == "PIM":
        dataset_train: WhaleDataset = PartImageNetDataset(pim_path,mode='train')
        dataset_val: WhaleDataset = PartImageNetDataset(pim_path,mode='val')


    elif dataset == "CUB":
        train_transforms = transforms.Compose([
            transforms.Resize(size=height),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1),
            transforms.RandomCrop(size=height),
            transforms.ToTensor(),
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(size=height),
            transforms.CenterCrop(size=height),
            transforms.ToTensor(),
        ])
        dataset_train = CUBDataset(cub_path, split=0.9, mode='train', height=height, transform=train_transforms)
        dataset_val = CUBDataset(cub_path, split=0.9, mode='val', train_samples=dataset_train.trainsamples, height=height, transform=test_transforms)

        # don't use for now
        # dataset_test = CUBDataset(cub_path, mode='test', transform=test_transforms)


    train_batch = 20
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=train_batch, shuffle=True, num_workers=4)

    test_batch = 8
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=test_batch, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=test_batch, shuffle=False, num_workers=4)

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    number_epochs = 50
    model_name: str = f'{experiment}.pt'
    model_name_init: str = f'{experiment}.pt'
    warm_start: bool = False
    do_only_test: bool = False

    num_landmarks: int = 10

    weights = ResNet101_Weights.DEFAULT
    basenet = resnet101(weights=weights)

    if dataset=="WHALE":
        num_cls = 2000
    elif dataset=="CUB":
        num_cls = 200
    elif dataset=="PIM":
        num_cls = 160

    # net = Net(basenet, num_classes=num_cls)
    net = NewLandmarkNet(basenet, num_landmarks, num_classes=num_cls, height=height)

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
    scratch_layers = ["fc_class_landmarks"]
    # finer_layers = []
    # finer_layers = ["fc_landmarks"]
    #TODO try attention scratch
    finer_layers = ["fc_landmarks", "fc_class", "fc", "mha", "fc_class_attention", "class_intoken"]
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
    # optimizer = torch.optim.SGD(
    #     [{'params': scratch_parameters, 'lr': 1e-2},
    #      {'params': finetune_parameters, 'lr': 5e-4},
    #      ], weight_decay=5e-4, momentum=0.9)
    # optimizer = torch.optim.Adam(
    #             [{'params': list(net.parameters())[:-2], 'lr': 1e-4},
    #              {'params': list(net.parameters())[-2:-1], 'lr': 1e-3},
    #              {'params': list(net.parameters())[-1:], 'lr': 1e-2}])
    # optimizer = torch.optim.SGD(
    #             [{'params': list(net.parameters())[:-2], 'lr': 1e-4},
    #              {'params': list(net.parameters())[-2:-1], 'lr': 1e-3},
    #              {'params': list(net.parameters())[-1:], 'lr': 1e-2}],
    #             weight_decay=5e-4, momentum=0.9)
    # optimizer2 = torch.optim.Adam(
    #     [{'params': list(net.parameters())[-4:-2], 'lr': 1e-4},
    #      {'params': list(net.parameters())[-2:-1], 'lr': 1e-3}]
    # )
    # for i in list(net.parameters()):
    #     print(i.shape)
    # optimizer = torch.optim.Adam([{'params': list(net.parameters())[:-1], 'lr': 1e-4},
    #      {'params': list(net.parameters())[-1:], 'lr': 1e-2}],
    #     weight_decay=5e-4)

    loss_fn = torch.nn.CrossEntropyLoss()
    baselr = 1e-4
    # try different schedulers
    # STEPLR
    optimizer = torch.optim.Adam(
            [{'params': scratch_parameters, 'lr': baselr*100},
             {'params': finer_parameters, 'lr': baselr*10},
             {'params': finetune_parameters, 'lr': baselr},
             ])
    # optimizer = torch.optim.Adam(
    #     [{'params': list(net.parameters())[:-2], 'lr': baselr},
    #      {'params': list(net.parameters())[-2:-1], 'lr': baselr},
    #      {'params': list(net.parameters())[-1:], 'lr': baselr * 100}])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    for epoch in range(epoch_leftoff, number_epochs):
        if epoch == 3:
            optimizer = torch.optim.SGD(
                [{'params': scratch_parameters, 'lr': baselr*1000},
                 {'params': finer_parameters, 'lr': baselr*100},
                 {'params': finetune_parameters, 'lr': baselr*10}],
                weight_decay=5e-4, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

        if not do_only_test:
            if all_losses:
                net, all_losses = train(net, optimizer, train_loader, device, model_name, epoch, 0, loss_fn, all_losses)
            else:
                net, all_losses = train(net, optimizer, train_loader, device, model_name, epoch, epoch_leftoff, loss_fn)
            scheduler.step()
            print(f'Validation accuracy in epoch {epoch}:')
            validation(device, net, val_loader, epoch, do_only_test)
            torch.cuda.empty_cache()
        # Validation
        else:
            print('Validation accuracy with saved network:')
            validation(device, net, val_loader, epoch, do_only_test)

if __name__=="__main__":
    main()
