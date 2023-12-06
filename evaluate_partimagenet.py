"""
In large part from: https://github.com/subhc/unsup-parts/
"""

# pytorch & misc
from datasets import PartImageNetDataset
from nets import *
from torchvision.models import resnet101
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from lib import *
import torch.nn.functional as F
import argparse

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
        (inputs, _, landmarks) = next(iter_loader)
        # to device
        inputs, landmarks = inputs.cuda(), landmarks.cuda()

        # Used to filter out all pixels that have < 0.1 value for all GT parts
        background_landmark = torch.full(size=(1, 1, landmarks.shape[-2], landmarks.shape[-1]), fill_value=0.1).cuda()
        landmarks_full = torch.cat((landmarks, background_landmark), dim=1)

        # Check which part is most active per pixel
        landmarks_argmax = torch.argmax(landmarks_full, dim=1)
        landmarks_vec = landmarks_argmax.view(-1)

        with torch.no_grad():
            # generate assignment map
            _, maps, _ = net(inputs)
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
    parser = argparse.ArgumentParser(description='Evaluate PDiscoNet parts on PartImageNet')
    parser.add_argument('--model_path', help='Path to .pt file', required=True)
    parser.add_argument('--data_root', help='The directory containing partimagenet folder', required=True)
    parser.add_argument('--num_parts', help='Number of parts the model was trained with', required=True, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    args = parser.parse_args()
    # define data transformation (no crop)
    num_cls = 110
    # define dataset and loader
    eval_data = PartImageNetDataset(args.data_root + '/partimagenet', mode='test', get_masks=True, evaluate=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=False, drop_last=False)

    # load the net in eval mode
    basenet = resnet101()
    net = IndividualLandmarkNet(basenet, args.num_parts, num_classes=num_cls).cuda()
    checkpoint = torch.load(args.model_path)

    net.load_state_dict(checkpoint, strict=True)
    net.eval()

    nmi, ari = eval_nmi_ari(net, eval_loader)
    print('NMI between predicted and ground truth parts is %.2f' % nmi)
    print('ARI between predicted and ground truth parts is %.2f' % ari)
    print('Evaluation finished.')

if __name__ == '__main__':
    main()
