from lib import show_maps, landmark_coordinates, rotate_image, flip_image, get_epoch
from datasets import WhaleDataset, WhaleTripletDataset, PartImageNetDataset, PartImageNetTripletDataset, CUBDataset, CUBTripletDataset
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
experiment = "cub_conc_div4"
if not os.path.exists(f'./results_{experiment}'):
    os.mkdir(f'./results_{experiment}')
# Loss hyperparameters
l_max = 1

l_equiv = 1
# l_equiv = 0

l_conc = 0.25

l_orth = 1
# l_orth = 0

l_class = 1

l_comp = 1
# l_comp = 0

# dataset = "WHALE"
# dataset = "PIM"
dataset = "CUB"

def train(net: torch.nn.Module, train_loader: torch.utils.data.DataLoader, device: torch.device, do_baseline: bool, model_name: str,epoch: int, epoch_leftoff, all_losses: list = None):
    # Training
    if all_losses:
        running_loss, running_loss_conc, running_loss_mean, running_loss_max, running_loss_class, running_loss_equiv, running_loss_orth, running_loss_comp = all_losses
    elif not all_losses and epoch != 0:
        print('Please pass the losses of the previous epoch to the training function')
    triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
    classif_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [{'params': list(net.parameters())[0:-1], 'lr': 1e-4},
         {'params': list(net.parameters())[-1:], 'lr': 1e-2}])
    net.train()
    all_training_vectors = []
    all_training_labels = []
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    iter_loader = iter(train_loader)
    for i in range(len(train_loader)):
        sample = next(iter_loader)
        with torch.no_grad():
            anchor, _, _, _ = net(sample[0].to(device))
            all_training_vectors.append(anchor.cpu())
            all_training_labels.append(sample[3])
        ### DATA AUGMENTATION PROCEDURES
        # Data augmentation 1: Flip the positive and the anchor
        # Do not use classif loss, since it would look like a different individual
        do_class = 1
        if np.random.rand() > 0.5:
            sample[0] = sample[0].flip(-1)
            sample[1] = sample[1].flip(-1)
            do_class = 0
        # Data augmentation 2: Flip the negative
        if np.random.rand() > 0.5:
            sample[2] = sample[2].flip(-1)

        # Data augmentation 3: Substitute the negative with the flipped positive or anchor
        if np.random.rand() > 0.9:
            if np.random.rand() > 0.5:
                sample[2] = sample[0].flip(-1)
            else:
                sample[2] = sample[1].flip(-1)
        
        # Data augmentation 4: random transform anchor, positive and negative
        angle = np.random.randn() * 0.1
        scale = np.random.rand() * 0.2 + 0.9
        # Anchor
        sample[0] = torchvision.transforms.functional.affine(sample[0],
                                                             angle=angle * 180 / np.pi,
                                                             translate=[0,0],
                                                             scale=scale,
                                                             shear=0)
        angle = np.random.randn() * 0.1
        scale = np.random.rand() * 0.2 + 0.9
        # Positive
        sample[1] = torchvision.transforms.functional.affine(sample[1],
                                                             angle=angle * 180 / np.pi,
                                                             translate=[0,0],
                                                             scale=scale,
                                                             shear=0)
        angle = np.random.randn() * 0.1
        scale = np.random.rand() * 0.2 + 0.9
        # Negative
        sample[2] = torchvision.transforms.functional.affine(sample[2],
                                                             angle=angle * 180 / np.pi,
                                                             translate=[0,0],
                                                             scale=scale,
                                                             shear=0)

        ### FORWARD PASS OF TRIPLET
        anchor, maps, scores_anchor, feature_tensor = net(sample[0].to(device))
        positive, _, scores_pos, _ = net(sample[1].to(device))
        negative, _, _, _ = net(sample[2].to(device))

        ### FORWARD PASS OF ROTATED IMAGES
        rot_img, rot_angle = rotate_image([0, 90, 180, 270], sample[0])
        if rot_angle == 0:
            flip_img, is_flipped = flip_image(rot_img, 1)
        else:
            flip_img, is_flipped = flip_image(rot_img, 0.5)
        _, equiv_map, _, _ = net(flip_img.to(device))

        if do_baseline:
            loss = triplet_loss(anchor, positive, negative)
            loss_class = classif_loss(scores_anchor, sample[3].to(
                device)) / 2 + classif_loss(scores_pos,
                                            sample[3].to(device)) / 2
            total_loss = loss + 10 * loss_class * do_class
            loss_conc = total_loss.detach() * 0
            loss_max = total_loss.detach() * 0
            loss_mean = total_loss.detach() * 0
            loss_equiv = total_loss.detach() * 0
            loss_orth = total_loss.detach() * 0
            loss_comp = total_loss.detach() * 0
        else:
            # Keep track of average distances between postives and negatives
            net.avg_dist_pos.data = net.avg_dist_pos.data * 0.95 + (
                    (anchor.detach() - positive.detach()) ** 2).mean(
                0).sum(0).sqrt() * 0.05
            net.avg_dist_neg.data = net.avg_dist_neg.data * 0.95 + (
                    (anchor.detach() - negative.detach()) ** 2).mean(
                0).sum(0).sqrt() * 0.05
            
            loss = 0  # triplet_loss((anchor).mean(2),(positive).mean(2),(negative).mean(2))
            # Classification loss for anchor and positive samples
            loss_class = classif_loss(scores_anchor.mean(-1), sample[3].to(
                device)) / 2 + classif_loss(scores_pos.mean(-1), sample[3].to(device)) / 2
            
            # Triplet loss for each triplet of landmarks
            for lm in range(anchor.shape[2]):
                loss += triplet_loss((anchor[:, :, lm]),
                                     (positive[:, :, lm]),
                                     (negative[:, :, lm])) / (
                            (anchor.shape[2] - 1))
                # loss_class += (classif_loss(scores_anchor[:,:,lm],sample[3].to(device))/2 + classif_loss(scores_pos[:,:,lm],sample[3].to(device))/2)/((anchor.shape[2]-1))
            
            # Classification loss using random subsets of landmarks
            for drops in range(0):
                # dropout_mask = (torch.rand(1,1,scores_anchor.shape[-1])>np.random.rand()*0.5).float().to(device)
                dropout_mask = (torch.rand(1, 1, scores_anchor.shape[-1]) > 0.5).float().to(device)
                d = 1 / (dropout_mask.mean() + 1e-6)
                loss_class += classif_loss(
                    (dropout_mask * scores_anchor).mean(-1) * d,
                    sample[3].to(device)) / 10
                loss_class += classif_loss(
                    (dropout_mask * scores_pos).mean(-1) * d,
                    sample[3].to(device)) / 10
            # Get landmark coordinates
            loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps, device)

            # Concentration loss
            loss_conc_x = (loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) ** 2
            loss_conc_y = (loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y) ** 2
            loss_conc = ((loss_conc_x + loss_conc_y)) * maps
            loss_conc = loss_conc[:, 0:-1, :, :].mean()

            loss_max = maps.max(-1)[0].max(-1)[0].mean()
            loss_max = 1 - loss_max

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
            _, _, _, comp_featuretensor = net(masked_imgs)
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

            # total_loss = loss + loss_conc + loss_max + loss_class * do_class + loss_equiv + loss_orth + loss_comp
            total_loss = loss + loss_conc + loss_max + loss_class * do_class
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch == epoch_leftoff and i == 0:
            running_loss = loss.item()
            running_loss_conc = loss_conc.item()
            running_loss_mean = loss_mean.item()
            running_loss_max = loss_max.item()
            running_loss_class = loss_class.item()
            running_loss_equiv = loss_equiv.item()
            running_loss_orth = loss_orth.item()
            running_loss_comp = loss_comp.item()

        else:
            # noinspection PyUnboundLocalVariable
            running_loss = 0.99 * running_loss + 0.01 * loss.item()
            # noinspection PyUnboundLocalVariable
            running_loss_conc = 0.99 * running_loss_conc + 0.01 * loss_conc.item()
            # noinspection PyUnboundLocalVariable
            running_loss_mean = 0.99 * running_loss_mean + 0.01 * loss_mean.item()
            # noinspection PyUnboundLocalVariable
            running_loss_max = 0.99 * running_loss_max + 0.01 * loss_max.item()
            # noinspection PyUnboundLocalVariable
            running_loss_class = 0.99 * running_loss_class + 0.01 * loss_class.item()

            running_loss_equiv = 0.99 * running_loss_equiv + 0.01 * loss_equiv.item()

            running_loss_orth = 0.99 * running_loss_orth + 0.01 * loss_orth.item()

            running_loss_comp = 0.99 * running_loss_comp + 0.01 * loss_comp.item()
            # running_loss_masked_diff = 0.99*running_loss_masked_diff + 0.01*loss_masked_diff.item()
        pbar.set_description(
            "T: %.3f, Cnc: %.3f, M: %.3f, Cls: %.3f, Eq: %.3f, Or: %.3f, Cmp: %.3f" % (
                running_loss, running_loss_conc,
                running_loss_max, running_loss_class, running_loss_equiv, running_loss_orth, running_loss_comp))

        pbar.update()
    pbar.close()
    torch.save(net.cpu().state_dict(), model_name)
    all_losses = running_loss, running_loss_conc, running_loss_mean, running_loss_max, running_loss_class, running_loss_equiv, running_loss_orth, running_loss_comp
    with open(f'./results_{experiment}/res.txt', 'a') as fopen:
        fopen.write(f'Epoch: {epoch}\n')
        fopen.write("T: %.3f, Cnc: %.3f, M: %.3f, Cls: %.3f, Eq: %.3f, Or: %.3f, Cmp: %.3f\n" % (
                running_loss, running_loss_conc,
                running_loss_max, running_loss_class, running_loss_equiv, running_loss_orth, running_loss_comp))
    return net, all_losses

def validation(device: torch.device, do_baseline: bool, net: torch.nn.Module, val_loader: torch.utils.data.DataLoader, epoch):
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
    for i, sample in enumerate(pbar):
        feat, maps, scores, feature_tensor = net(sample[0].to(device))
        scores = scores.detach().cpu()
        all_scores.append(scores)
        lab = sample[1]
        all_labels.append(lab)

        if do_baseline:
            for j in range(scores.shape[0]):
                sorted_scores, sorted_indeces = scores[j, :].softmax(0).sort(
                    descending=True)
                topk_class.append(list(sorted_indeces).index(lab[j]))
                top_class.append(sorted_indeces[0])
                diff_to_second.append(
                    float(sorted_scores[0] - sorted_scores[1]))
        else:
            for j in range(scores.shape[0]):
                sorted_scores, sorted_indeces = scores[j, :, :].mean(
                    -1).softmax(0).sort(descending=True)
                topk_class.append(list(sorted_indeces).index(lab[j]))
                top_class.append(sorted_indeces[0])
                diff_to_second.append(
                    float(sorted_scores[0] - sorted_scores[1]))
            if topk_lm_class is None:
                topk_lm_class = []
                class_lm = []
                for lm in range(feat.shape[2]):
                    topk_lm_class.append([])
                    class_lm.append([])
            for lm in range(feat.shape[2]):
                for j in range(scores.shape[0]):
                    class_lm[lm].append(
                        int(scores[j, :, lm].argmax().cpu().numpy()))
                    topk_lm_class[lm].append(
                        list(scores[j, :, lm].sort(descending=True)[1]).index(
                            lab[j]))
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

            if np.random.random() < 0.05:
                show_maps(sample[0], maps, loc_x, loc_y, epoch, experiment)

    top5acc = str((np.array(topk_class)<5).mean())

    print(top5acc)
    with open(f'results_{experiment}/res.txt', 'a') as fopen:
        fopen.write(top5acc + "\n")
    pbar.close()

def main():
    print(torch.cuda.is_available())

    whale_path: str = "./datasets/happyWhale"
    pim_path: str = "./datasets/pim"
    cub_path: str = "./datasets/cub/CUB_200_2011"

    if dataset == "WHALE":
        dataset_train: WhaleDataset = WhaleDataset(whale_path, mode='train')
        dataset_val: WhaleDataset = WhaleDataset(whale_path, mode='val')
        dataset_full: WhaleDataset = WhaleDataset(whale_path, mode='no_set', minimum_images=0,
                                    alt_data_path='Teds_OSM')
        dataset_train_triplet: WhaleTripletDataset = WhaleTripletDataset(dataset_train)

        batch_size: int = 12
        train_loader: DataLoader[Any] = torch.utils.data.DataLoader(dataset=dataset_train_triplet,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
        val_loader: DataLoader[Any] = torch.utils.data.DataLoader(dataset=dataset_val,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=4)
    elif dataset == "PIM":
        dataset_train: WhaleDataset = PartImageNetDataset(pim_path,mode='train')
        dataset_val: WhaleDataset = PartImageNetDataset(pim_path,mode='val')

        dataset_train_triplet: PartImageNetTripletDataset = PartImageNetTripletDataset(dataset_train)

        batch_size: int = 12
        train_loader: DataLoader[Any] = torch.utils.data.DataLoader(dataset=dataset_train_triplet,
                                                   batch_size=batch_size, shuffle=True,
                                                   num_workers=4)
        val_loader: DataLoader[Any] = torch.utils.data.DataLoader(dataset=dataset_val,
                                                 batch_size=batch_size, shuffle=False,
                                                 num_workers=4)

    elif dataset == "CUB":
        np.random.seed(1)
        dataset_train = CUBDataset(cub_path, mode='train')
        dataset_val = CUBDataset(cub_path, mode='val', train_samples=dataset_train.trainsamples)
        dataset_train_triplet = CUBTripletDataset(dataset_train)

        # don't use for now
        dataset_test = CUBDataset(cub_path, mode='test')



        batch_size = 12
        train_loader: DataLoader[Any] = torch.utils.data.DataLoader(dataset=dataset_train_triplet,
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

    do_baseline: bool = False
    num_landmarks: int = 10

    basenet: ResNet = torchvision.models.resnet18(pretrained=True)
    net: Union[Net,LandmarkNet]

    if dataset=="WHALE":
        num_cls = 2000
    elif dataset=="CUB":
        num_cls = 250
    elif dataset=="PIM":
        num_cls = None

    if do_baseline:
        net = Net(basenet, num_classes=num_cls)
    else:
        net = LandmarkNet(basenet, num_landmarks, num_classes=num_cls)

    if warm_start:
        net.load_state_dict(torch.load(model_name_init), strict=False)
        epoch_leftoff = get_epoch(experiment) + 1
    else:
        epoch_leftoff = 0
    net.to(device)

    if do_only_test:
        number_epochs = 1

    all_losses = []
    for epoch in range(epoch_leftoff, number_epochs):
        if not do_only_test:
            if all_losses:
                net, all_losses = train(net, train_loader, device, do_baseline, model_name,epoch, 0, all_losses)
            else:
                net, all_losses = train(net, train_loader, device, do_baseline, model_name, epoch, epoch_leftoff)
            print(f'Validation accuracy in epoch {epoch}:')
            validation(device, do_baseline, net, val_loader, epoch)
        # Validation
        else:
            print('Validation accuracy with saved network:')
            validation(device, do_baseline, net, val_loader, epoch)

if __name__=="__main__":
    main()
