from lib import show_maps, landmark_coordinates, get_epoch, rigid_transform
from dataset import CUBDataset
from nets import IndividualLandmarkNet

import os

import numpy as np
import torch
import torch.multiprocessing
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torchvision.models import resnet101, ResNet101_Weights
from tqdm import tqdm

import torch.nn.functional as F
import argparse

# to avoid error "too many files open"
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()


def train(net, optimizer, train_loader, device, model_name, epoch,
          epoch_leftoff, loss_fn, loss_hyperparams, all_losses=None):
    """
    Model trainer, saves losses to file
    Parameters
    ----------
    net: torch.nn.Module
        The model to train
    optimizer: torch.optim
        Optimizer used for training
    train_loader: torch.utils.data.DataLoader
        Data loader for the training set
    device: torch.device
        The device on which the network is trained
    model_name: str
        The name of the model, used to save results of model
    epoch: int
        Current epoch, used to save results of model
    epoch_leftoff: int
        Starting epoch of the training function, used if a training run was
        stopped at e.g. epoch 10 and then later continued from there
    loss_fn: torch loss function
        Loss function used for backpropagation
    loss_hyperparams: dict of {loss: hyperparam} as {str: float}
        Indicates, per loss, its hyperparameter
    all_losses: [float]
        The list of all running losses, used to display (not backprop)
    Returns
    ----------
    net: torch.nn.Module
        The model, trained for another epoch
    all_losses: [float]
        The list of all running losses, used to display
    """
    # Training
    if all_losses:
        running_loss_conc, running_loss_max, running_loss_class_lnd, \
        running_loss_equiv, running_loss_orth, \
        running_loss_class_att = all_losses
    elif not all_losses and epoch != 0:
        print(
            'Please pass the losses of the previous epoch to the training function')
    net.train()
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    iter_loader = iter(train_loader)
    top_class_att = []
    top_class_lnd = []
    l_class_lnd = loss_hyperparams['l_class_lnd']
    l_class_att = loss_hyperparams['l_class_att']
    l_max = loss_hyperparams['l_max']
    l_conc = loss_hyperparams['l_conc']
    l_orth = loss_hyperparams['l_orth']
    l_equiv = loss_hyperparams['l_equiv']
    for i in range(len(train_loader)):
        sample = next(iter_loader)
        lab = sample[1].to(device)
        anchor, maps, scores, features, classif = net(sample[0].to(device))

        # Forward pass of transformed images
        angle = np.random.rand()*180-90
        translate = list(np.int32(np.floor(np.random.rand(2)*100-50)))
        scale = np.random.rand()*0.6+0.8
        transf_img = rigid_transform(sample[0], angle, translate, scale,
                                     invert=False)
        _, equiv_map, _, _, _ = net(transf_img.to(device))

        # Classification loss for landmarks by themselves, and for
        # the attention layer
        loss_class_landmarks = loss_fn(scores[:, :, 0:-1].mean(-1), lab).mean()
        loss_class_landmarks = loss_class_landmarks * l_class_lnd
        loss_class_attention = loss_fn(classif, lab).mean()
        loss_class_attention = loss_class_attention * l_class_att

        for j in range(scores.shape[0]):
            probs_lnd = scores[j, :, :-1].mean(-1).softmax(dim=0).detach().cpu()
            preds_lnd = torch.argmax(probs_lnd, dim=-1).detach().cpu()
            probs_att = classif[j, :].softmax(dim=0).detach().cpu()
            preds_att = torch.argmax(probs_att, dim=-1).detach().cpu()
            top_class_lnd.append(1 if preds_lnd == lab[j].detach().cpu() else 0)
            top_class_att.append(1 if preds_att == lab[j].detach().cpu() else 0)

        # Get landmark coordinates
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps, device)

        # Concentration loss
        loss_conc_x = ((loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
        loss_conc_y = ((loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
        loss_conc = (loss_conc_x + loss_conc_y) * maps
        loss_conc = (loss_conc[:, 0:-1, :, :].mean()) * l_conc

        # Max/presence loss
        loss_max = torch.nn.functional.avg_pool2d(
            maps[:, :, 2:-2, 2:-2], 3, stride=1)\
                .max(-1)[0].max(-1)[0].max(0)[0].mean()
        loss_max = (1 - loss_max) * l_max

        # Orthogonality loss
        normed_feature = torch.nn.functional.normalize(anchor, dim=1)
        similarity = torch.matmul(normed_feature.permute(0, 2, 1),
                                  normed_feature)
        similarity = torch.sub(similarity, torch.eye(
            net.num_landmarks + 1).to(device))
        orth_loss = torch.mean(torch.square(similarity))
        loss_orth = orth_loss * l_orth

        # Equivariance loss: calculate rotated landmarks distance
        translate = [(t*maps.shape[-1]/sample[0].shape[-1]) for t in translate]
        rot_back = rigid_transform(equiv_map, angle, translate,
                                   scale, invert=True)
        num_elements_per_map = maps.shape[-2] * maps.shape[-1]
        orig_attmap_vector = torch.reshape(maps[:, :-1, :, :],
                                           (-1, net.num_landmarks,
                                            num_elements_per_map))
        transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :],
                                             (-1, net.num_landmarks,
                                              num_elements_per_map))
        cos_sim_equiv = F.cosine_similarity(orig_attmap_vector,
                                            transf_attmap_vector, -1)
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
    torch.save(net.cpu().state_dict(), f'{model_name}.pt')
    all_losses = running_loss_conc, running_loss_max, \
                 running_loss_class_lnd, running_loss_equiv, \
                 running_loss_orth, running_loss_class_att
    with open(f'../results_{model_name}/res.txt', 'a') as fopen:
        fopen.write(f'Epoch: {epoch}\n')
        fopen.write(
            "Cnc: %.3f, M: %.3f, Lnd: %.3f, Eq: %.3f, Or: %.3f, Att: %.3f\n" % (
                running_loss_conc,
                running_loss_max, running_loss_class_lnd,
                running_loss_equiv, running_loss_orth,
                running_loss_class_att))
        fopen.write(
            f"Att training top 1: {str(np.mean(np.array(top_class_att)))}\n")
        fopen.write(
            f"Lnd training top 1: {str(np.mean(np.array(top_class_lnd)))}\n")
    return net, all_losses


def validation(device, net, val_loader, epoch, only_test, model_name, save_maps=True):
    """
    Calculates validation accuracy for trained model, saves it to file
    Parameters
    ----------
    device: torch.device
        The device on which the network is loaded
    net: torch.nn.Module
        The model to evaluate
    val_loader: torch.utils.data.DataLoader
        Data loader for the validation set
    epoch: int
        Current epoch, used to save results
    only_test: bool
        Whether this is a run where the model is only being evaluated
    model_name: str
        Name of the model, used to save results
    save_maps: bool
        Whether to save the attention maps
    """
    net.eval()
    net.to(device)
    pbar = tqdm(val_loader, position=0, leave=True)
    top_class_lnd = []
    top_class_att = []
    all_scores = []
    all_labels = []
    all_maxes = torch.Tensor().to(device)
    for i, sample in enumerate(pbar):
        anchor, maps, scores, feature_tensor, classif = net(sample[0].to(device))
        scores = scores.detach().cpu()
        all_scores.append(scores)
        lab = sample[1]
        all_labels.append(lab)

        for j in range(scores.shape[0]):
            probs_lnd = scores[j, :, :-1].mean(-1).softmax(dim=0).cpu()
            preds_lnd = torch.argmax(probs_lnd, dim=-1).cpu()
            probs_att = classif[j, :].softmax(dim=0).cpu()
            preds_att = torch.argmax(probs_att, dim=-1).cpu()
            top_class_lnd.append(1 if preds_lnd == lab[j].cpu() else 0)
            top_class_att.append(1 if preds_att == lab[j].cpu() else 0)

        map_max = maps.max(-1)[0].max(-1)[0][:, :-1].detach()
        all_maxes = torch.cat((all_maxes, map_max), 0)

        # Saving the attention maps
        if save_maps:
            grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]), torch.arange(maps.shape[3]))
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)
            map_sums = maps.sum(3).sum(2).detach()
            maps_x = grid_x * maps
            maps_y = grid_y * maps
            loc_x = maps_x.sum(3).sum(2) / map_sums
            loc_y = maps_y.sum(3).sum(2) / map_sums
            if np.random.random() < 0.01:
                show_maps(sample[0], maps, loc_x, loc_y, epoch, model_name)

    top1acc = str(np.mean(np.array(top_class_att)))
    top1acclnd = str(np.mean(np.array(top_class_lnd)))
    print(f"Att validation top 1: {top1acc}")
    print(f"Lnd validation top 1: {top1acclnd}")
    if not only_test:
        with open(f'../results_{model_name}/res.txt', 'a') as fopen:
            fopen.write(f"Att validation top 1: {top1acc} \n")
            fopen.write(f"Lnd validation top 1: {top1acclnd} \n")
    pbar.close()

def main():
    parser = argparse.ArgumentParser(
        description='PDiscoNet on CUB'
    )
    parser.add_argument('--model_name', help='used to train a new model',
                        required=True)
    parser.add_argument('--data_path',
                        help='directory that contains cub files, must'
                             'contain folder "./images"', required=True)
    parser.add_argument('--num_parts', help='number of parts to predict',
                        default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=15, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--epochs', default=28, type=int)
    parser.add_argument('--pretrained_model_name', default='',
                        help='used to load pretrained model')
    parser.add_argument('--save_maps', default=True, type=bool)
    parser.add_argument('--warm_start', default=False,
                        help='Whether to use a pretrained PDiscoNet', type=bool)
    parser.add_argument('--only_test', default=False,
                        help='Whether to only eval the model', type=bool)
    args = parser.parse_args()

    if not os.path.exists(f'../results_{args.model_name}'):
        os.mkdir(f'../results_{args.model_name}')
    if torch.cuda.is_available():
        print("Using GPU to train.")
    else:
        print("Using CPU to train.")

    np.random.seed(1)
    train_transforms = transforms.Compose([
        transforms.Resize(size=args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1),
        transforms.RandomAffine(degrees=90,translate=(0.2,0.2),scale=(0.8,1.2)),
        transforms.RandomCrop(args.image_size),
        transforms.ToTensor()
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(size=args.image_size),
        transforms.CenterCrop(size=args.image_size),
        transforms.ToTensor()
    ])
    dataset_train = CUBDataset(args.data_path, split=0.9, mode='train',
                               height=args.image_size, transform=train_transforms)
    dataset_test = CUBDataset(args.data_path, mode='test',
                              transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)

    test_batch = 8
    val_loader = torch.utils.data.DataLoader(dataset=dataset_test,
                                             batch_size=test_batch,
                                             shuffle=False, num_workers=4)

    weights = ResNet101_Weights.DEFAULT
    basenet = resnet101(weights=weights)
    num_cls = 200

    net = IndividualLandmarkNet(basenet, args.num_parts, num_classes=num_cls)

    if args.warm_start:
        net.load_state_dict(torch.load(args.pretrained_model_name) + '.pt',
                            strict=False)
        epoch_leftoff = get_epoch(args.model_name) + 1
    else:
        epoch_leftoff = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    if args.only_test:
        epoch_leftoff = 0
        args.epochs = 1

    all_losses = []

    scratch_layers = ["modulation"]
    finer_layers = ["fc_class_landmarks", "fc_class_attention"]
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

    loss_hyperparams = {'l_class_lnd': 1, 'l_class_att': 1, 'l_max': 1,
                        'l_equiv': 1, 'l_conc': 1000, 'l_orth': 1}
    # STEPLR
    optimizer = torch.optim.Adam(
        [{'params': scratch_parameters, 'lr': args.lr * 100},
         {'params': finer_parameters, 'lr': args.lr * 10},
         {'params': finetune_parameters, 'lr': args.lr},
         ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.5)
    for epoch in range(epoch_leftoff, args.epochs):
        if not args.only_test:
            if all_losses:
                net, all_losses = train(net, optimizer, train_loader, device,
                                        args.model_name, epoch, 0, loss_fn,
                                        loss_hyperparams, all_losses)
            else:
                net, all_losses = train(net, optimizer, train_loader, device,
                                        args.model_name, epoch, epoch_leftoff,
                                        loss_fn, loss_hyperparams)
            scheduler.step()
            print(f'Validation accuracy in epoch {epoch}:')
            validation(device, net, val_loader, epoch, args.only_test,
                       args.model_name, save_maps=args.save_maps)
            torch.cuda.empty_cache()
        # Validation
        else:
            print('Validation accuracy with saved network:')
            validation(device, net, val_loader, epoch, args.only_test,
                       args.model_name, save_maps=args.save_maps)


if __name__ == "__main__":
    main()
