"""
From: https://github.com/zxhuang1698/interpretability-by-parts/blob/master/src/cub200/eval_interp.py

"""


# pytorch & misc
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import *
from nets import *
from torchvision.models import resnet101
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from lib import *

# number of attributes and landmark annotations
num_classes = 200
num_landmarks = 15

def cut_borders(inputs, landmarks, bbox):
    """
    Cut the right and bottom border of the image to make both height
    and width multiples of 16 (downsampling factor of the model).
    Note that different input size leads to different padding patterns,
    meaning that the coordinates of feature maps of different inputs
    are not fully aligned.
    Parameters
    ----------
    inputs: torch.cuda.FloatTensor, [1, 3, height, width]
        Tensor for the input image.
    landmarks: torch.cuda.FloatTensor, [1, num_landmarks, 4]
        Tensor for landmark annotations.
    bbox: torch.cuda.FloatTensor, [1, 5]
        Tensor for the bounding box annotations.
    Returns
    ----------
    inputs_cropped: torch.cuda.FloatTensor, [1, 3, new_height, new_width]
        Tensor for the cropped input image.
    landmarks_cropped: torch.cuda.FloatTensor, [1, num_landmarks, 4]
        Tensor for landmark annotations.
    bbox_cropped: torch.cuda.FloatTensor, [1, 5]
        Tensor for the cropped bounding box annotation.
    """
    # row and col check, x means row num here
    height = inputs.shape[2]
    width = inputs.shape[3]

    # calculate border width and height, effectively center cropping the image
    border_h = height - 448
    border_w = width - 448

    # calculate the new shape
    new_height = height - border_h
    new_width = width - border_w

    # cut the inputs
    inputs_cropped = inputs[:, :, (border_h//2):new_height+(border_h//2), (border_w//2):new_width+(border_w//2)]

    # transform the landmarks correspondingly
    landmarks_cropped = landmarks.squeeze(0)
    for i in range(num_landmarks):
        if abs(landmarks_cropped[i][-1]) > 1e-5:
            landmarks_cropped[i][-3] -= (border_w//2) # column shift
            landmarks_cropped[i][-2] -= (border_h//2) # row shift

            # remove the landmarks if unlucky (never really stepped in for test set of cub200)
            if landmarks_cropped[i][-3] < 0 or landmarks_cropped[i][-3] >= new_width or \
                landmarks_cropped[i][-2] < 0 or landmarks_cropped[i][-2] >= new_height:
                landmarks_cropped[i][-3] = 0
                landmarks_cropped[i][-2] = 0
                landmarks_cropped[i][-1] = 0

    landmarks_cropped = landmarks_cropped.unsqueeze(0)

    # transform the bounding box correspondingly
    bbox_cropped = bbox
    bbox_cropped[0, 1] = bbox_cropped[0, 1] - (border_w//2)
    bbox_cropped[0, 2] = bbox_cropped[0, 2] - (border_h//2)
    bbox_cropped[0, 1].clamp_(min=0)
    bbox_cropped[0, 2].clamp_(min=0)
    bbox_cropped[0, 3].clamp_(max=new_width)
    bbox_cropped[0, 4].clamp_(max=new_height)

    return inputs_cropped, landmarks_cropped, bbox_cropped

def calc_center(assignment, num_parts):
    """
    Calculate the geometric centers for assignment maps.
    Parameters
    ----------
    assignment: torch.cuda.FloatTensor, [batch_size, num_parts, height, width]
        Soft assignment map before upsampling.
    num_parts: int, number of object parts.
    Returns
    ----------
    centers: torch.cuda.FloatTensor, [batch_size, num_parts * 2]
        Geometric centers for assignment maps with order
        (col_1, row_1, ..., col_K, row_K)
    """
    # row and col check, x means row num here
    batch_size = assignment.shape[0]
    nparts = assignment.shape[1]
    height = assignment.shape[2]
    width = assignment.shape[3]

    # assertions
    assert nparts == num_parts


    # generate the location map
    col_map = torch.arange(1, width+1).view(1, 1, 1, width).expand(batch_size, nparts, height, width).float().cuda()
    row_map = torch.arange(1, height+1).view(1, 1, height, 1).expand(batch_size, nparts, height, width).float().cuda()

    # multiply the location map with the soft assignment map
    col_weighted = (col_map * assignment).view(batch_size, nparts, -1)
    row_weighted = (row_map * assignment).view(batch_size, nparts, -1)

    # sum of assignment as the denominator
    denominator = torch.sum(assignment.view(batch_size, nparts, -1), dim=2) + 1e-8

    # calculate the weighted average of location maps as centers
    col_mean = torch.sum(col_weighted, dim=2) / denominator
    row_mean = torch.sum(row_weighted, dim=2) / denominator

    # prepare the centers for return
    col_centers = col_mean.unsqueeze(2)
    row_centers = row_mean.unsqueeze(2)

    # N * K * 1 -> N * K * 2 -> N * (K * 2)
    centers = torch.cat([col_centers, row_centers], dim=2).view(batch_size, nparts * 2)

    # upsample to the image resolution (256, 256)
    centers = centers * 16

    return centers

def create_centers(data_loader, model, num_parts):
    """
    Generate the center coordinates as tensor for the current model.
    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Data loader for the current data split.
    model: pytorch model object
        Model that generates assignment maps.
    num_parts: int, number of object parts
    Returns
    ----------
    centers_tensor: torch.cuda.FloatTensor, [data_size, num_parts * 2]
        Geometric centers for assignment maps of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    annos_tensor: torch.cuda.FloatTensor, [batch_size, num_landmarks * 2]
        Landmark coordinate annotations of the whole dataset.
        The coordinate order is (col_1, row_1, ..., col_K, row_K)
    """
    # tensor for collecting centers, labels, existence masks
    centers_collection = []
    annos_collection = []
    masks_collection = []
    active_parts_collection = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    # bbox shape: [N, 5]
    for i, (input_raw, _, landmarks_raw, bbox_raw) in enumerate(data_loader):

        # to device
        input_raw = input_raw.cuda()
        landmarks_raw = landmarks_raw.cuda()
        bbox_raw = bbox_raw.cuda()

        # cut the input and transform the landmark
        inputs, landmarks_full, bbox = cut_borders(input_raw, landmarks_raw, bbox_raw)

        # gather the landmark annotations, center outputs and existence masks
        with torch.no_grad():

            # generate assignment map
            _, assignment, _, _, _ = model(inputs)
            assignment = assignment[:, :-1, :, :]
            maxes = assignment.max(-1)[0].max(-1)[0]
            active_parts = torch.where(maxes > 1e-2, 1, 0).unsqueeze(-1).expand(-1, -1, 2)

            # calculate the center coordinates of shape [N, num_parts, 2]
            centers = calc_center(assignment, num_parts)
            centers = centers.contiguous().view(centers.shape[0], num_parts, 2)

            # extract the landmark and existence mask, [N, num_landmarks, 2]
            landmarks = landmarks_full[:, :, -3:-1]
            masks = landmarks_full[:, :, -1].unsqueeze(2).expand_as(landmarks)

            # normalize the coordinates with the bounding boxes
            bbox = bbox.unsqueeze(2)
            centers[:, :, 0] = (centers[:, :, 0] - bbox[:, 1]) / bbox[:, 3]
            centers[:, :, 1] = (centers[:, :, 1] - bbox[:, 2]) / bbox[:, 4]
            landmarks[:, :, 0] = (landmarks[:, :, 0] - bbox[:, 1]) / bbox[:, 3]
            landmarks[:, :, 1] = (landmarks[:, :, 1] - bbox[:, 2]) / bbox[:, 4]


            # collect the centers, annotations and masks
            centers_collection.append(centers)
            annos_collection.append(landmarks)
            masks_collection.append(masks)
            active_parts_collection.append(active_parts)

    # list into tensors
    centers_tensor = torch.cat(centers_collection, dim=0)
    annos_tensor = torch.cat(annos_collection, dim=0)
    masks_tensor = torch.cat(masks_collection, dim=0)
    active_tensor = torch.cat(active_parts_collection, dim=0)

    # reshape the tensors
    centers_tensor = centers_tensor.contiguous().view(centers_tensor.shape[0], num_parts * 2)
    annos_tensor = annos_tensor.contiguous().view(centers_tensor.shape[0], num_landmarks * 2)
    masks_tensor = masks_tensor.contiguous().view(centers_tensor.shape[0], num_landmarks * 2)
    active_tensor = active_tensor.contiguous().view(active_tensor.shape[0], num_parts * 2)

    return centers_tensor, annos_tensor, masks_tensor, active_tensor

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

def get_nmi_inputs(data_loader, model):
    all_nmi_preds_w_bg = []
    all_nmi_gts = []

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    # bbox shape: [N, 5]
    for i, (input_raw, _, landmarks_raw, bbox_raw) in enumerate(data_loader):

        # to device
        input_raw = input_raw.cuda()
        landmarks_raw = landmarks_raw.cuda()
        bbox_raw = bbox_raw.cuda()

        # cut the input and transform the landmark
        inputs, landmarks_full, bbox = cut_borders(input_raw, landmarks_raw, bbox_raw)

        with torch.no_grad():

            # generate assignment map
            maps = model(inputs)[1]
            part_name_mat_w_bg = F.interpolate(maps, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

            if i == 1000:
                plt.imshow(inputs.permute(0,2,3,1).cpu().numpy().squeeze())
                plt.show()
                plt.imshow(part_name_mat_w_bg.argmax(1).cpu().numpy().squeeze())
    #                 plt.show()
    #                 print(landmarks_full.shape)
                plt.scatter(landmarks_full[0, :, 1].detach().cpu(), landmarks_full[0, :, 2].detach().cpu(), color='red')
                plt.show()



            # extract the landmark and existence mask, [N, num_landmarks, 2]
            visible = landmarks_full[:, :, 3] > 0.5
            points = landmarks_full[:, :, 1:3].unsqueeze(2).clone()

            points[:, :, :, 0] /= inputs.shape[-1]  # W
            points[:, :, :, 1] /= inputs.shape[-2]  # H
            assert points.min() > -1e-7 and points.max() < 1+1e-7
            points = points * 2 - 1

            pred_parts_loc_w_bg = F.grid_sample(part_name_mat_w_bg.float(), points, mode='nearest', align_corners=False)
            pred_parts_loc_w_bg = torch.argmax(pred_parts_loc_w_bg, dim=1).squeeze(2)
            pred_parts_loc_w_bg = pred_parts_loc_w_bg[visible]
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.cpu().numpy())

            gt_parts_loc = torch.arange(landmarks_full.shape[1]).unsqueeze(0).repeat(landmarks_full.shape[0], 1)
            gt_parts_loc = gt_parts_loc[visible]
            all_nmi_gts.append(gt_parts_loc.cpu().numpy())


    all_nmi_preds_w_bg = np.concatenate(all_nmi_preds_w_bg, axis=0)
    all_nmi_gts = np.concatenate(all_nmi_gts, axis=0)

    return all_nmi_gts, all_nmi_preds_w_bg


def eval_all():
    # define data transformation (no crop)
    nparts = 15
    num_cls = 200
    height = 448
    data_transforms = transforms.Compose([
        transforms.Resize(size=(height)),
        # transforms.CenterCrop(size=height),
        transforms.ToTensor(),
    ])
    cub_path = "./datasets/cub"
    # define dataset and loader
    fit_data = CUB200(cub_path, train=True, transform=data_transforms)
    eval_data = CUB200(cub_path, train=False, transform=data_transforms)
    fit_loader = torch.utils.data.DataLoader(
        fit_data, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, drop_last=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, drop_last=False)
    # load the model in eval mode
    basenet = resnet101()
    model = NewLandmarkNet(basenet, nparts, num_classes=num_cls, height=height).cuda()

    checkpoint = torch.load("15parts_fulldataset.pt")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    nmi_gts, nmi_preds_w_bg = get_nmi_inputs(eval_loader, model)
    print(np.unique(nmi_preds_w_bg))
    nmi = normalized_mutual_info_score(nmi_gts, nmi_preds_w_bg) * 100
    ari = adjusted_rand_score(nmi_gts, nmi_preds_w_bg) * 100
    print(nmi)
    print(ari)
    return nmi, ari


def main():

    # define data transformation (no crop)
    data_transforms = transforms.Compose([
        transforms.Resize(size=(448)),
        # transforms.CenterCrop(size=448), #keep?
        transforms.ToTensor()
        ])

    # define dataset and loader
    cub_path = "./datasets/cub"
    height = 448
    num_cls = 200
    nparts = 8

    fit_data = CUB200(cub_path, train=True, transform=data_transforms)
    eval_data = CUB200(cub_path, train=False, transform=data_transforms)
    fit_loader = torch.utils.data.DataLoader(
        fit_data, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, drop_last=False)
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, drop_last=False)
    # load the model in eval mode
    basenet = resnet101()
    model = NewLandmarkNet(basenet, nparts, num_classes=num_cls, height=height).cuda()

    checkpoint = torch.load("8parts_fulldataset_standardized_nodetach.pt")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # convert the assignment to centers for both splits
    print('Evaluating the model for the whole data split...')
    fit_centers, fit_annos, fit_masks, fit_active_centers = create_centers(fit_loader, model, nparts)
    eval_centers, eval_annos, eval_masks, eval_active_centers = create_centers(eval_loader, model, nparts)

    # fit the linear regressor with sklearn
    # normalized assignment center coordinates -> normalized landmark coordinate annotations
    print('=> fitting and evaluating the regressor')
    error = 0
    n_valid_samples = 0

    # different landmarks have different masks
    for i in range(num_landmarks):

        # get the valid indices for the current landmark
        fit_masks_np = fit_masks.cpu().numpy().astype(np.float64)
        eval_masks_np = eval_masks.cpu().numpy().astype(np.float64)
        fit_selection = (abs(fit_masks_np[:, i*2]) > 1e-5)
        eval_selection = (abs(eval_masks_np[:, i*2]) > 1e-5)

        # fit_centers = torch.where(fit_active_centers, fit_centers, 0)
        # eval_centers = torch.where(eval_active_centers, eval_centers, 0)

        # convert tensors to numpy (64 bit double)
        fit_centers_np = fit_centers.cpu().numpy().astype(np.float64)
        fit_annos_np = fit_annos.cpu().numpy().astype(np.float64)
        eval_centers_np = eval_centers.cpu().numpy().astype(np.float64)
        eval_annos_np = eval_annos.cpu().numpy().astype(np.float64)

        # select the current landmarks for both fit and eval set
        fit_annos_np = fit_annos_np[:, i*2:i*2+2]
        eval_annos_np = eval_annos_np[:, i*2:i*2+2]

        # remove invalid indices
        fit_centers_np = fit_centers_np[fit_selection]

        fit_annos_np = fit_annos_np[fit_selection]
        eval_centers_np = eval_centers_np[eval_selection]
        eval_annos_np = eval_annos_np[eval_selection]
        eval_data_size = eval_centers_np.shape[0]

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
        fit_centers_std = fit_centers_std[fit_active_centers]
        fit_centers_std = torch.where(fit_active_centers, fit_centers_std, 0)


        # define regressor without intercept and fit it
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(fit_centers_std, fit_annos_std)

        # standardize the centers on the evaluation split
        eval_centers_std = scaler_centers.transform(eval_centers_np)

        # Take only the active centers
        eval_centers_std = torch.where(eval_active_centers, eval_centers_std, 0)

        # regress the landmarks on the evaluation split
        eval_pred_std = regressor.predict(eval_centers_std)




        # unstandardize the prediction with StandardScaler for landmarks
        eval_pred = scaler_landmarks.inverse_transform(eval_pred_std)

        # calculate the error
        eval_pred = eval_pred.reshape((eval_data_size, 1, 2))
        eval_annos_np = eval_annos_np.reshape((eval_data_size, 1, 2))
        error += L2_distance(eval_pred, eval_annos_np) * eval_data_size
        n_valid_samples += eval_data_size

    error = error * 100 / n_valid_samples
    print('Mean L2 Distance on the test set is %.2f%%.' % error)
    print('Evaluation finished for model.')

if __name__ == '__main__':
    main()
