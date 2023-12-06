"""
Contains functions used for training and testing
"""


# Import statements
import torch
import numpy as np
from lib import rigid_transform, landmark_coordinates, save_maps
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function definitions
def conc_loss(centroid_x: torch.Tensor, centroid_y: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor,
              maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the concentration loss, which is the weighted sum of the squared distance of the landmark
    Parameters
    ----------
    centroid_x: torch.Tensor
        The x coordinates of the map centroids
    centroid_y: torch.Tensor
        The y coordinates of the map centroids
    grid_x: torch.Tensor
        The x coordinates of the grid
    grid_y: torch.Tensor
        The y coordinates of the grid
    maps: torch.Tensor
        The attention maps

    Returns
    -------
    loss_conc: torch.Tensor
        The concentration loss
    """
    spatial_var_x = ((centroid_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
    spatial_var_y = ((centroid_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
    spatial_var_weighted = (spatial_var_x + spatial_var_y) * maps
    loss_conc = spatial_var_weighted[:, 0:-1, :, :].mean()
    return loss_conc


def orth_loss(num_parts: int, landmark_features: torch.Tensor, device) -> torch.Tensor:
    """
    Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
    Parameters
    ----------
    num_parts: int
        The number of landmarks
    landmark_features: torch.Tensor, [batch_size, feature_dim, num_landmarks + 1 (background)]
        Tensor containing the feature vector for each part
    device: torch.device
        The device to use
    Returns
    -------
    loss_orth: torch.Tensor
        The orthogonality loss
    """
    normed_feature = torch.nn.functional.normalize(landmark_features, dim=1)
    similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
    similarity = torch.sub(similarity, torch.eye(num_parts + 1).to(device))
    loss_orth = torch.mean(torch.square(similarity))
    return loss_orth


def equiv_loss(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int) \
        -> torch.Tensor:
    """
    Calculates the equivariance loss, which we calculate from the cosine similarity between the original attention map
    and the inversely transformed attention map of a transformed image.
    Parameters
    ----------
    X: torch.Tensor
        The input image
    maps: torch.Tensor
        The attention maps
    net: torch.nn.Module
        The model
    device: torch.device
        The device to use
    num_parts: int
        The number of landmarks

    Returns
    -------
    loss_equiv: torch.Tensor
        The equivariance loss
    """
    # Forward pass
    angle = np.random.rand() * 180 - 90
    translate = list(np.int32(np.floor(np.random.rand(2) * 100 - 50)))
    scale = np.random.rand() * 0.6 + 0.8
    transf_img = rigid_transform(X, angle, translate, scale, invert=False)
    _, equiv_map, _ = net(transf_img.to(device))

    # Compare to original attention map, and penalise high difference
    translate = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    rot_back = rigid_transform(equiv_map, angle, translate, scale, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :], (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :], (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = F.cosine_similarity(orig_attmap_vector, transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    return loss_equiv

def train(net: torch.nn.Module, optimizer: torch.optim, train_loader: torch.utils.data.DataLoader,
          device: torch.device, epoch: int, epoch_leftoff: int, loss_fn: torch.nn.Module, loss_hyperparams: dict,
          writer: torch.utils.tensorboard.SummaryWriter, all_losses: [float] = None) -> (torch.nn.Module, [float]):
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
    epoch: int
        Current epoch, used for the running loss
    epoch_leftoff: int
        Starting epoch of the training function, used if a training run was
        stopped at e.g. epoch 10 and then later continued from there
    loss_fn: torch.nn.Module
        Loss function
    loss_hyperparams: dict
        Indicates, per loss, its hyperparameter
    writer: torch.utils.tensorboard.SummaryWriter
        The object to write performance metrics to
    all_losses: [float]
        The list of all running losses, used to display (not backprop)
    Returns
    ----------
    net: torch.nn.Module
        The model with updated weights
    all_losses: [float]
        The list of all running losses, used to display (not backprop)
    """
    # Training
    if all_losses:
        running_loss_conc, running_loss_pres, running_loss_class, running_loss_equiv, running_loss_orth = all_losses
    elif not all_losses and epoch != 0:
        print(
            'Please pass the losses of the previous epoch to the training function')
    net.train()
    pbar = tqdm(total=len(train_loader), position=0, leave=True)
    top_class = []
    l_class = loss_hyperparams['l_class']
    l_pres = loss_hyperparams['l_pres']
    l_conc = loss_hyperparams['l_conc']
    l_orth = loss_hyperparams['l_orth']
    l_equiv = loss_hyperparams['l_equiv']
    for i, (X, lab) in enumerate(train_loader):
        lab = lab.to(device)
        landmark_features, maps, scores = net(X.to(device))
        # Equivariance loss: calculate rotated landmarks distance
        loss_equiv = equiv_loss(X, maps, net, device, net.num_landmarks) * l_equiv

        # Classification loss
        loss_class = loss_fn(scores[:, :, 0:-1].mean(-1), lab).mean()
        loss_class = loss_class * l_class

        # Classification accuracy
        preds = scores[:, :, :-1].mean(-1).argmax(dim=1)
        top_class.append((preds == lab).float().mean().cpu())
        # for j in range(scores.shape[0]):
        #     probs = scores[j, :, :-1].mean(-1).softmax(dim=0).detach().cpu()
        #     preds = torch.argmax(probs, dim=-1).detach().cpu()
        #     top_class.append(1 if preds == lab[j].detach().cpu() else 0)

        # Get landmark coordinates
        loc_x, loc_y, grid_x, grid_y = landmark_coordinates(maps, device)

        # Concentration loss
        loss_conc = conc_loss(loc_x, loc_y, grid_x, grid_y, maps) * l_conc

        # Presence loss
        loss_pres = torch.nn.functional.avg_pool2d(maps[:, :, 2:-2, 2:-2], 3, stride=1).max(-1)[0].max(-1)[0].max(0)[0].mean()
        loss_pres = (1 - loss_pres) * l_pres

        # Orthogonality loss
        loss_orth = orth_loss(net.num_landmarks, landmark_features, device) * l_orth

        total_loss = loss_conc + loss_pres + loss_orth + loss_equiv + loss_class
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        if epoch == epoch_leftoff and i == 0:
            running_loss_conc = loss_conc.item()
            running_loss_pres = loss_pres.item()
            running_loss_class = loss_class.item()
            running_loss_equiv = loss_equiv.item()
            running_loss_orth = loss_orth.item()
        else:
            running_loss_conc = 0.99 * running_loss_conc + 0.01 * loss_conc.item()
            running_loss_pres = 0.99 * running_loss_pres + 0.01 * loss_pres.item()
            running_loss_class = 0.99 * running_loss_class + 0.01 * loss_class.item()
            running_loss_equiv = 0.99 * running_loss_equiv + 0.01 * loss_equiv.item()
            running_loss_orth = 0.99 * running_loss_orth + 0.01 * loss_orth.item()
        pbar.update()

    top1acc = np.mean(np.array(top_class))
    writer.add_scalar('Concentration loss', running_loss_conc, epoch)
    writer.add_scalar('Presence loss', running_loss_pres, epoch)
    writer.add_scalar('Classification loss', running_loss_class, epoch)
    writer.add_scalar('Equivariance loss', running_loss_equiv, epoch)
    writer.add_scalar('Orthogonality loss', running_loss_orth, epoch)
    writer.add_scalar('Training Accuracy', top1acc, epoch)

    pbar.close()
    all_losses = running_loss_conc, running_loss_pres, running_loss_class, running_loss_equiv, running_loss_orth
    writer.flush()
    return net, all_losses

def validation(device, net, val_loader, epoch, model_name, save_figures, writer):
    """
    Calculates validation accuracy for trained model, writes it to Tensorboard Summarywriter.
    Also saves figures with attention maps if save_figures is set to True.
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
    model_name: str
        Name of the model, used to save results
    save_figures: bool
        Whether to save the attention maps
    writer: torch.utils.tensorboard.SummaryWriter
        The object to write metrics to
    """
    net.eval()
    net.to(device)
    pbar = tqdm(val_loader, position=0, leave=True)
    top_class = []
    all_scores = []
    all_labels = []
    all_maxes = torch.Tensor().to(device)
    with torch.no_grad():
        for i, (X, y) in enumerate(tqdm(val_loader)):
            _, maps, scores = net(X.to(device))
            scores = scores.detach().cpu()
            all_scores.append(scores)
            lab = y
            all_labels.append(lab)

            for j in range(scores.shape[0]):
                probs = scores[j, :, :-1].mean(-1).softmax(dim=0).cpu()
                preds = torch.argmax(probs, dim=-1).cpu()
                top_class.append(1 if preds == lab[j].cpu() else 0)

            map_max = maps.max(-1)[0].max(-1)[0][:, :-1].detach()
            all_maxes = torch.cat((all_maxes, map_max), 0)

            # Saving the attention maps
            if save_figures and i % 100 == 0:
                save_maps(X, maps, epoch, model_name, device)

    top1acc = np.mean(np.array(top_class))
    writer.add_scalar('Validation Accuracy', top1acc, epoch)
    pbar.close()
    writer.flush()

if __name__ == "__main__":
    pass
