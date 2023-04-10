"""
From: https://github.com/zxhuang1698/interpretability-by-parts/blob/master/src/cub200/eval_interp.py
"""

# pytorch & misc
from dataset import *
from nets import *
from torchvision.models import resnet101
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from lib import *
import torch.nn.functional as F

# number of attributes and landmark annotations
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
    iter_loader = iter(data_loader)

    # iterating the data loader, landmarks shape: [N, num_landmarks, 4], column first
    # bbox shape: [N, 5]
    for i in range(len(data_loader)):
        print(i)
        (input_raw, _, landmarks_raw) = next(iter_loader)
        # to device
        input_raw = input_raw.cuda()
        landmarks_raw = landmarks_raw.cuda()

        # cut the input and transform the landmark
        inputs, landmarks_full = input_raw, landmarks_raw

        # Used to filter out all pixels that have < 0.1 value for all GT parts
        background_landmark = torch.full(size=(1, 1, landmarks_full.shape[-2], landmarks_full.shape[-1]), fill_value=0.1).cuda()
        landmarks_full = torch.cat((landmarks_full, background_landmark), dim=1)

        # Check which part is most active per pixel
        landmarks_argmax = torch.argmax(landmarks_full, dim=1)
        landmarks_vec = landmarks_argmax.view(-1)

        with torch.no_grad():
            # generate assignment map
            maps = net(inputs)[1]
            part_name_mat_w_bg = F.interpolate(maps, size=inputs.shape[-2:], mode='bilinear', align_corners=False)

            pred_parts_loc_w_bg = torch.argmax(part_name_mat_w_bg, dim=1)
            pred_parts_loc_w_bg = pred_parts_loc_w_bg.view(-1)
            all_nmi_preds_w_bg.append(pred_parts_loc_w_bg.cpu().numpy())
            all_nmi_gts.append(landmarks_vec.cpu().numpy())

    nmi_preds = np.concatenate(all_nmi_preds_w_bg, axis=0)
    nmi_gts = np.concatenate(all_nmi_gts, axis=0)

    nmi = normalized_mutual_info_score(nmi_gts, nmi_preds) * 100
    ari = adjusted_rand_score(nmi_gts, nmi_preds) * 100
    return nmi, ari

def main():
    # define data transformation (no crop)
    nparts = 50
    num_cls = 110
    height = 256
    data_transforms = transforms.Compose([
        transforms.Resize(size=height),
        transforms.ToTensor(),
    ])
    pim_path = "../datasets/pim"
    # define dataset and loader
    eval_data = PartImageNetDataset(pim_path,
                        mode='test', transform=data_transforms, get_masks=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, drop_last=False)

    # load the net in eval mode
    basenet = resnet101()
    net = IndividualLandmarkNet(basenet, nparts, num_classes=num_cls).cuda()
    checkpoint = torch.load("../archive/PIM_50parts_all_losses_normal_ABLATION_INDIVIDUAL.pt")
    net.load_state_dict(checkpoint, strict=True)
    net.eval()

    nmi, ari = eval_nmi_ari(net, eval_loader)
    print('NMI between predicted and ground truth parts is %.2f' % nmi)
    print('ARI between predicted and ground truth parts is %.2f' % ari)
    print('Evaluation finished.')

if __name__ == '__main__':
    main()
