"""
Adapted from: https://github.com/zxhuang1698/interpretability-by-parts/blob/master/src/cub200/eval_interp.py
and https://github.com/subhc/unsup-parts/blob/dba2ced195e48bde9eb33cd3ab7134a1ffe8ad75/evaluation/CUB%20eval.ipynb
"""

# pytorch & misc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as transforms
from datasets import CelebA
from nets import *
from torchvision.models import resnet101
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from lib import *
import torch.nn.functional as F
import argparse

# Naming used: 'landmarks' are ground truth parts, 'parts' are predicted parts
num_landmarks = 5


def create_centers(data_loader, net, num_parts):
    """
    Generate the center coordinates as tensor for the current net.
    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Data loader for the current data split.
    net: torch.nn.Module
        Net that generates assignment maps.
    num_parts: int
        Number of predicted parts
    Returns
    ----------
    centers_tensor: torch.cuda.FloatTensor, [data_size, num_parts * 2]
        Geometric centers for assignment maps of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    annos_tensor: torch.cuda.FloatTensor, [batch_size, num_landmarks * 2]
        Landmark coordinate annotations of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    eyedists_tensor: torch.cuda.FloatTensor, [data_size, 1]
        Distance between the eyes for every image
    active_tensor: torch.cuda.FloatTensor, [data_size, num_parts * 2]
        Contains for each predicted part whether it's active or not. If a part
        has a maximum attention map value of > 0.5 somewhere in the image,
        we denote it as active.
    """
    # tensor for collecting centers, labels, existence masks
    centers_collection = []
    annos_collection = []
    eyedists_collection = []
    active_parts_collection = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    for i, (inputs, _, landmarks) in enumerate(data_loader):

        # to device
        inputs = inputs.cuda()
        landmarks = landmarks.cuda()

        # gather the landmark annotations, center outputs and existence masks
        with torch.no_grad():

            # generate assignment map
            _, maps, _, = net(inputs)
            maps = maps[:, :-1, :, :]
            maxes = maps.max(-1)[0].max(-1)[0]
            active_parts = torch.where(maxes > 0.5, 1, 0).unsqueeze(-1).expand(-1, -1, 2)

            # calculate the center coordinates of shape [N, num_parts, 2]
            loc_x, loc_y, grid_x, grid_y = landmark_coordinates(
                maps.cpu(), "cpu")
            x_centroid = loc_x * inputs.shape[-2]/maps.shape[-2]
            y_centroid = loc_y * inputs.shape[-1]/maps.shape[-1]
            centers = torch.stack((x_centroid, y_centroid), dim=-1)

            # collect the centers and annotations, active parts
            centers_collection.append(centers)
            annos_collection.append(landmarks)
            active_parts_collection.append(active_parts)

            # calculate and collect the normalization factor
            col_dist = landmarks[:, 1, 0] - landmarks[:, 0, 0]
            row_dist = landmarks[:, 1, 1] - landmarks[:, 0, 1]
            eye_dist = (col_dist * col_dist + row_dist * row_dist).sqrt()
            eyedists_collection.append(eye_dist)

    # list into tensors
    centers_tensor = torch.cat(centers_collection, dim=0).view(-1, num_parts*2).cuda()
    annos_tensor = torch.cat(annos_collection, dim=0)
    eyedists_tensor = torch.cat(eyedists_collection, dim=0)
    active_tensor = torch.cat(active_parts_collection, dim=0)

    # reshape the tensors
    annos_tensor = annos_tensor.contiguous().view(annos_tensor.shape[0], num_landmarks * 2)
    eyedists_tensor = eyedists_tensor.contiguous().unsqueeze(1)
    active_tensor = active_tensor.contiguous().view(active_tensor.shape[0], num_parts * 2)
    return centers_tensor, annos_tensor, eyedists_tensor, active_tensor

def L2_distance(prediction, annotation):
    """
    Average L2 distance of two numpy arrays.
    Parameters
    ----------
    prediction: np.array, [data_size, 1, 2]
        Landmark prediction.
    annotation: np.array, [data_size, 1, 2]
        Landmark annotation.
    Returns
    ----------
    error: float
        Average L2 distance between prediction and annotation.
    """
    diff_sq = (prediction - annotation) * (prediction - annotation)
    L2_dists = np.sqrt(np.sum(diff_sq, axis=2))
    error = np.mean(L2_dists)
    return error

def eval_nmi_ari(net, data_loader):
    """
    Get Normalized Mutual Information, Adjusted Rand Index for given method
    Parameters
    ----------
    net: torch.nn.Module
        The trained net to evaluate
    data_loader: torch.utils.data.DataLoader
        The dataset to evaluate
    Returns
    ----------
    nmi: float
        Normalized Mutual Information between predicted parts and gt parts as %
    ari: float
        Adjusted Rand Index between predicted parts and gt parts as %
    """
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    # bbox shape: [N, 5]
    for i, (inputs, _, landmarks) in enumerate(data_loader):

        # to device
        inputs = inputs.cuda()
        landmarks = landmarks.cuda()

        with torch.no_grad():

            # generate assignment map
            maps = net(inputs)[1]
            part_name_mat_w_bg = F.interpolate(maps,
                size=inputs.shape[-2:], mode='bilinear', align_corners=False)

            # extract the landmark and existence mask, [N, num_landmarks, 2]
            points = landmarks.unsqueeze(2).clone()
            points[:, :, :, 0] /= inputs.shape[-1]  # W
            points[:, :, :, 1] /= inputs.shape[-2]  # H
            assert points.min() > -1e-7 and points.max() < 1+1e-7
            points = points * 2 - 1

            pred_parts_loc_w_bg = F.grid_sample(part_name_mat_w_bg.float(),
                                points, mode='nearest', align_corners=False)
            pred_parts_loc_w_bg = torch.argmax(pred_parts_loc_w_bg,
                                               dim=1).squeeze(2)
            pred_parts_loc_w_bg = pred_parts_loc_w_bg[0]
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.cpu().numpy())

            gt_parts_loc = torch.arange(landmarks.shape[1]).unsqueeze(0).repeat(landmarks.shape[0], 1)
            gt_parts_loc = gt_parts_loc[0]
            all_nmi_gts.append(gt_parts_loc.cpu().numpy())

    nmi_preds = np.concatenate(all_nmi_preds_w_bg, axis=0)
    nmi_gts = np.concatenate(all_nmi_gts, axis=0)
    nmi = normalized_mutual_info_score(nmi_gts, nmi_preds) * 100
    ari = adjusted_rand_score(nmi_gts, nmi_preds) * 100

    return nmi, ari


def eval_kpr(net, fit_loader, eval_loader, nparts):
    """
    Evaluate keypoint regression for given method
    Parameters
    ----------
    net: torch.nn.Module
        The trained net to evaluate
    fit_loader: torch.utils.data.DataLoader
        The dataset on which to train the keypoint regression
    eval_loader: torch.utils.data.DataLoader
        The dataset on which to evaluate the keypoint regression
    nparts: int
        Number of predicted parts
    Returns
    ----------
    kpr: float
        Keypoint regression between centroids of predicted parts and gt parts
    """
    # convert the assignment to centers for both splits
    print('Evaluating the keypoint regression for the whole data split...')
    fit_centers, fit_annos, fit_eyedists, fit_active_centers = create_centers(fit_loader, net, nparts)
    eval_centers, eval_annos, eval_eyedists, eval_active_centers = create_centers(eval_loader, net, nparts)
    eval_data_size = eval_centers.shape[0]

    # normalize the centers to make sure every face image has unit eye distance
    fit_centers, fit_annos = fit_centers / fit_eyedists, fit_annos / fit_eyedists
    eval_centers, eval_annos = eval_centers / eval_eyedists, eval_annos / eval_eyedists

    # fit the linear regressor with sklearn
    # normalized assignment center coordinates -> normalized landmark coordinate annotations
    print('=> fitting and evaluating the regressor')

    # convert tensors to numpy (64 bit double)
    fit_centers_np = fit_centers.cpu().numpy().astype(np.float64)
    fit_annos_np = fit_annos.cpu().numpy().astype(np.float64)
    eval_centers_np = eval_centers.cpu().numpy().astype(np.float64)
    eval_annos_np = eval_annos.cpu().numpy().astype(np.float64)

    # data standardization
    scaler_centers = StandardScaler()
    scaler_landmarks = StandardScaler()

    # fit the StandardScaler with the fitting split
    scaler_centers.fit(fit_centers_np)
    scaler_landmarks.fit(fit_annos_np)

    # stardardize the fitting split
    fit_centers_std = scaler_centers.transform(fit_centers_np)
    fit_annos_std = scaler_landmarks.transform(fit_annos_np)

    # Take only the active centers
    fit_centers_std = np.where(fit_active_centers.cpu().numpy(),
                               fit_centers_std, 0)

    # define regressor without intercept and fit it
    regressor = LinearRegression(fit_intercept=False)
    regressor.fit(fit_centers_std, fit_annos_std)

    # standardize the centers on the evaluation split
    eval_centers_std = scaler_centers.transform(eval_centers_np)

    # Take only the active centers
    eval_centers_std = np.where(eval_active_centers.cpu().numpy(),
                                eval_centers_std, 0)

    # regress the landmarks on the evaluation split
    eval_pred_std = regressor.predict(eval_centers_std)

    # unstandardize the prediction with StandardScaler for landmarks
    eval_pred = scaler_landmarks.inverse_transform(eval_pred_std)

    # calculate the error
    eval_pred = eval_pred.reshape((eval_data_size, num_landmarks, 2))
    eval_annos = eval_annos_np.reshape((eval_data_size, num_landmarks, 2))
    error = L2_distance(eval_pred, eval_annos) * 100

    return error


def main():
    parser = argparse.ArgumentParser(description='Evaluate PDiscoNet parts on CUB')
    parser.add_argument('--model_path', help='Path to .pt file', required=True)
    parser.add_argument('--data_root', help='The directory containing the celeba folder', required=True)
    parser.add_argument('--num_parts', help='Number of parts the model was trained with', required=True, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    args = parser.parse_args()
    # define data transformation (no crop)
    num_cls = 10177
    # define dataset and loader
    eval_data = CelebA(args.data_root + '/celeba', split='eval', percentage=0.3, evaluate=True)
    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False,
                                              drop_last=False)
    # load the net in eval mode
    basenet = resnet101()
    net = IndividualLandmarkNet(basenet, args.num_parts, num_classes=num_cls).cuda()
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint, strict=True)
    net.eval()

    # Calculate keypoint regression error
    fit_data = CelebA(args.data_root + '/celeba', split='fit', percentage=0.3, evaluate=True)
    fit_loader = torch.utils.data.DataLoader(fit_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=False,
                                             drop_last=False)
    kpr = eval_kpr(net, fit_loader, eval_loader, args.num_parts)
    print('Mean keypoint regression error on the test set is %.2f%%.' % kpr)

    # Calculate NMI and ARI
    nmi, ari = eval_nmi_ari(net, eval_loader)
    print('NMI between predicted and ground truth parts is %.2f' % nmi)
    print('ARI between predicted and ground truth parts is %.2f' % ari)
    print('Evaluation finished.')


if __name__ == '__main__':
    main()
