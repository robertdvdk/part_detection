from lib import get_epoch
from datasets import PartImageNetDataset, CUBDataset, CelebA
from nets import IndividualLandmarkNet
import os
import argparse
import numpy as np
import torch
import torch.multiprocessing
from torch.utils.data import Dataset
from torchvision.models import resnet101, ResNet101_Weights
import json
from torch.utils.tensorboard import SummaryWriter
from train import train, validation

def main():
    parser = argparse.ArgumentParser(description='PDiscoNet')
    parser.add_argument('--model_name', help='Name under which the model will be saved', required=True)
    parser.add_argument('--data_root',
                    help='directory that contains the celeba, cub, or partimagenet folder', required=True)
    parser.add_argument('--dataset', help='The dataset to use. Choose celeba, cub, or partimagenet.', required=True)
    parser.add_argument('--num_parts', help='number of parts to predict', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=448, type=int) # 256 for celeba, 448 for cub,  224 for partimagenet
    parser.add_argument('--epochs', default=20, type=int) # 15 for celeba, 28 for cub, 20 for partimagenet
    parser.add_argument('--pretrained_model_path', default='', help='If you want to load a pretrained model,'
                        'specify the path to the model here.')
    parser.add_argument('--save_figures', default=False,
                        help='Whether to save the attention maps to png', action='store_true')
    parser.add_argument('--only_test', default=False, action='store_true', help='Whether to only test the model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter(log_dir=f'{args.dataset}/{args.model_name}')
    writer.add_text('Dataset', args.dataset.lower())
    writer.add_text('Device', str(device))
    writer.add_text('Learning rate', str(args.lr))
    writer.add_text('Batch size', str(args.batch_size))
    writer.add_text('Epochs', str(args.epochs))
    writer.add_text('Number of parts', str(args.num_parts))

    with open(f'{args.dataset}/{args.model_name}.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    np.random.seed(1)
    data_path = args.data_root + '/' + args.dataset.lower()
    if args.dataset.lower() == 'celeba':
        dataset_train = CelebA(data_path, 'train', 0.3)
        dataset_val = CelebA(data_path, 'val', 0.3)
        num_cls = 10177
    elif args.dataset.lower() == 'cub':
        dataset_train = CUBDataset(data_path + '/CUB_200_2011', split=1.0, mode='train', image_size=args.image_size)
        dataset_val = CUBDataset(data_path + '/CUB_200_2011', mode='test',
                                 train_samples=dataset_train.trainsamples, image_size=args.image_size)
        num_cls = 200
    elif args.dataset.lower() == 'partimagenet':
        dataset_train = PartImageNetDataset(data_path, mode='train')
        dataset_val = PartImageNetDataset(data_path, mode='test')
        num_cls = 110
    else:
        raise RuntimeError("Choose celeba, cub, or partimagenet as dataset")

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                               num_workers=4)

    test_batch = 8
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=test_batch, shuffle=True, num_workers=4)

    weights = ResNet101_Weights.DEFAULT
    basenet = resnet101(weights=weights)

    net = IndividualLandmarkNet(basenet, args.num_parts, num_classes=num_cls)

    if args.pretrained_model_path:
        if not os.path.exists(f'./results_{args.model_name}'):
            os.mkdir(f'./results_{args.model_name}')
        net.load_state_dict(torch.load(args.pretrained_model_path))

    net.to(device)
    epoch_leftoff = 0

    if args.only_test:
        args.epochs = 1

    all_losses = []

    high_lr_layers = ["modulation"]
    med_lr_layers = ["fc_class_landmarks"]

    # First entry contains parameters with high lr, second with medium lr, third with low lr
    param_dict = [{'params': [], 'lr': args.lr * 100},
                  {'params': [], 'lr': args.lr * 10},
                  {'params' : [], 'lr': args.lr}]
    for name, p in net.named_parameters():
        layer_name = name.split('.')[0]
        if layer_name in high_lr_layers:
            param_dict[0]['params'].append(p)
        elif layer_name in med_lr_layers:
            param_dict[1]['params'].append(p)
        else:
            param_dict[2]['params'].append(p)
    optimizer = torch.optim.Adam(params=param_dict)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss_hyperparams = {'l_class': 2, 'l_pres': 1, 'l_equiv': 1, 'l_conc': 1000, 'l_orth': 1}

    if args.dataset.lower() == 'celeba':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, 0.5)
    elif args.dataset.lower() == 'cub':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    elif args.dataset.lower() == 'partimagenet':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)

    for epoch in range(epoch_leftoff, args.epochs):
        if not args.only_test:
            if all_losses:
                net, all_losses = train(net, optimizer, train_loader, device, epoch, 0, loss_fn,
                                        loss_hyperparams, writer, all_losses)
            else:
                net, all_losses = train(net, optimizer, train_loader, device, epoch, epoch_leftoff,
                                        loss_fn, loss_hyperparams, writer)
            scheduler.step()
            print(f'Validation accuracy in epoch {epoch}:')
            validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
            torch.cuda.empty_cache()
        # Validation
        else:
            print('Validation accuracy with saved network:')
            validation(device, net, val_loader, epoch, args.model_name, args.save_figures, writer)
        torch.save(net.state_dict(), f'./{args.dataset}/{args.model_name}.pt')
    writer.close()

if __name__ == "__main__":
    main()