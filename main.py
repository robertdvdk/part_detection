# import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional

from torch.utils.data import Dataset
from tqdm import tqdm
import os

from datasets import WhaleDataset, WhaleTripletDataset
from nets import Net, LandmarkNet


def train(net, train_loader, device, do_baseline, model_name,epoch):
    # Training
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
        if i % 100 == -1:
            train_loader.dataset.height_list[0] = np.random.randint(200,
                                                                    300)
            train_loader.dataset.height_list[1] = np.random.randint(200,
                                                                    300)
            train_loader.dataset.height_list[2] = np.random.randint(200,
                                                                    300)
            iter_loader = iter(train_loader)
        sample = next(iter_loader)
        with torch.no_grad():
            anchor, _, _, _ = net(sample[0].to(device))
            all_training_vectors.append(anchor.cpu())
            all_training_labels.append(sample[3])

        # Flip the positive and the anchor
        do_class = 1
        if np.random.rand() > 0.5:
            sample[0] = sample[0].flip(-1)
            sample[1] = sample[1].flip(-1)
            do_class = 0
        # Flip the negative
        if np.random.rand() > 0.5:
            sample[2] = sample[2].flip(-1)

        # Substitute the negative with the flipped positive or anchor
        if np.random.rand() > 0.9:
            if np.random.rand() > 0.5:
                sample[2] = sample[0].flip(-1)
            else:
                sample[2] = sample[1].flip(-1)

        angle = np.random.randn() * 0.1
        scale = np.random.rand() * 0.2 + 0.9
        sample[0] = torchvision.transforms.functional.affine(sample[0],
                                                             angle=angle * 180 / np.pi,
                                                             translate=[0,
                                                                        0],
                                                             scale=scale,
                                                             shear=0)
        angle = np.random.randn() * 0.1
        scale = np.random.rand() * 0.2 + 0.9
        sample[1] = torchvision.transforms.functional.affine(sample[1],
                                                             angle=angle * 180 / np.pi,
                                                             translate=[0,
                                                                        0],
                                                             scale=scale,
                                                             shear=0)
        angle = np.random.randn() * 0.1
        scale = np.random.rand() * 0.2 + 0.9
        sample[2] = torchvision.transforms.functional.affine(sample[2],
                                                             angle=angle * 180 / np.pi,
                                                             translate=[0,
                                                                        0],
                                                             scale=scale,
                                                             shear=0)

        anchor, maps, scores_anchor, feature_tensor = net(
            sample[0].to(device))
        positive, _, scores_pos, _ = net(sample[1].to(device))
        negative, _, _, _ = net(sample[2].to(device))

        if not do_baseline:
            net.avg_dist_pos.data = net.avg_dist_pos.data * 0.95 + (
                    (anchor.detach() - positive.detach()) ** 2).mean(
                0).sum(0).sqrt() * 0.05
            net.avg_dist_neg.data = net.avg_dist_neg.data * 0.95 + (
                    (anchor.detach() - negative.detach()) ** 2).mean(
                0).sum(0).sqrt() * 0.05

        if do_baseline:
            loss = triplet_loss(anchor, positive, negative)
            loss_class = classif_loss(scores_anchor, sample[3].to(
                device)) / 2 + classif_loss(scores_pos,
                                            sample[3].to(device)) / 2
            total_loss = loss + 10 * loss_class * do_class
            loss_conc = total_loss.detach() * 0
            loss_max = total_loss.detach() * 0
            loss_mean = total_loss.detach() * 0
        else:
            loss = 0  # triplet_loss((anchor).mean(2),(positive).mean(2),(negative).mean(2))
            loss_class = classif_loss(scores_anchor.mean(-1), sample[3].to(
                device)) / 2 + classif_loss(scores_pos.mean(-1),
                                            sample[3].to(device)) / 2
            for lm in range(anchor.shape[2]):
                loss += triplet_loss((anchor[:, :, lm]),
                                     (positive[:, :, lm]),
                                     (negative[:, :, lm])) / (
                            (anchor.shape[2] - 1))
                # loss_class += (classif_loss(scores_anchor[:,:,lm],sample[3].to(device))/2 + classif_loss(scores_pos[:,:,lm],sample[3].to(device))/2)/((anchor.shape[2]-1))

            for drops in range(0):
                # dropout_mask = (torch.rand(1,1,scores_anchor.shape[-1])>np.random.rand()*0.5).float().to(device)
                dropout_mask = (torch.rand(1, 1, scores_anchor.shape[
                    -1]) > 0.5).float().to(device)
                d = 1 / (dropout_mask.mean() + 1e-6)
                loss_class += classif_loss(
                    (dropout_mask * scores_anchor).mean(-1) * d,
                    sample[3].to(device)) / 10
                loss_class += classif_loss(
                    (dropout_mask * scores_pos).mean(-1) * d,
                    sample[3].to(device)) / 10
            # Get landmark coordinates
            grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                            torch.arange(maps.shape[3]))
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

            map_sums = maps.sum(3).sum(2).detach()
            maps_x = grid_x * maps
            maps_y = grid_y * maps
            loc_x = maps_x.sum(3).sum(2) / map_sums
            loc_y = maps_y.sum(3).sum(2) / map_sums

            # Concentration loss
            loss_conc_x = (loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x) ** 2
            loss_conc_y = (loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y) ** 2
            loss_conc = ((loss_conc_x + loss_conc_y)) * maps
            loss_conc = loss_conc[:, 0:-1, :, :].mean()

            loss_max = maps.max(-1)[0].max(-1)[0].mean()
            loss_max = 1 - loss_max

            loss_mean = maps[:, 0:-1, :, :].mean()
            total_loss = loss + 1 * loss_conc + 0 * loss_mean + 1 * loss_max + 1 * loss_class * do_class  # + 1*loss_masked_diff

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch == 0 and i == 0:
            running_loss = loss.item()
            running_loss_conc = loss_conc.item()
            running_loss_mean = loss_mean.item()
            running_loss_max = loss_max.item()
            running_loss_class = loss_class.item()
            # running_loss_masked_diff = loss_masked_diff.item()
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
            # running_loss_masked_diff = 0.99*running_loss_masked_diff + 0.01*loss_masked_diff.item()
        pbar.set_description(
            "Training loss: %f, Conc: %f, Mean: %f, Max: %f, Class: %f" % (
                running_loss, running_loss_conc, running_loss_mean,
                running_loss_max, running_loss_class))
        pbar.update()
    pbar.close()
    torch.save(net.cpu().state_dict(), model_name)
    return net



def main():
    print(torch.cuda.is_available())
    if not os.path.exists('./happyWhale'):
        # try:
        os.mkdir('./happyWhale')
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files('humpback-whale-identification',
                                       path='./happyWhale/')

        import zipfile

        with zipfile.ZipFile('./happyWhale/humpback-whale-identification.zip', 'r') as zipref:
            zipref.extractall('./happyWhale/')
        # except Exception as e:
        #     os.rmdir('./happyWhale')
        #     print(e)
        #     raise RuntimeError("Unable to download Kaggle files! Please read README.md")


    data_path = "./happyWhale"

    dataset_train = WhaleDataset(data_path, mode='train')
    dataset_val = WhaleDataset(data_path, mode='val')
    dataset_full = WhaleDataset(data_path, mode='no_set', minimum_images=0,
                                alt_data_path='Teds_OSM')
    dataset_train_triplet = WhaleTripletDataset(dataset_train)

    batch_size = 12
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train_triplet,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val,
                                             batch_size=batch_size, shuffle=False,
                                             num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    number_epochs = 40
    model_name = 'landmarks_10_nodrop_4.pt'
    model_name_init = 'landmarks_10_nodrop_4.pt'
    warm_start = False
    do_only_test = True

    do_baseline = False
    num_landmarks = 10

    basenet = torchvision.models.resnet18(pretrained=True)
    if do_baseline:
        net = Net(basenet)
    else:
        net = LandmarkNet(basenet, num_landmarks)
    if warm_start:
        net.load_state_dict(torch.load(model_name_init), strict=False)
    net.to(device)



    if do_only_test:
        number_epochs = 1

    val_accs = []

    test_batch = []
    test_batch_labels = []
    # for id in test_list: # TODO what is test_list? -> probably list of whale ids in test set
    #     num = list(dataset_full.names).index(id)
    #     test_batch.append(torch.Tensor(dataset_full[num][0]).unsqueeze(0))
    #     test_batch_labels.append(dataset_full.unique_labels[dataset_full[num][1]])


    for epoch in range(number_epochs):
        if not do_only_test:
            net = train(net, train_loader, device, do_baseline, model_name,epoch)
        # Validation
        validation(device, do_baseline, net, val_loader)


def validation(device, do_baseline, net, val_loader):
    net.eval()
    net.to(device)
    pbar = tqdm(val_loader, position=0, leave=True)
    top_class = []
    names = []
    topk_class = []
    diff_to_second = []
    topk_lm_class = None
    class_lm = None
    all_scores = []
    all_labels = []
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
                    # topk_lm.append([])
                    topk_lm_class.append([])
                    class_lm.append([])
            for lm in range(feat.shape[2]):
                for j in range(scores.shape[0]):
                    # topk_lm[lm].append(list(sorted_labels[j,:]).index(lab[j]))
                    class_lm[lm].append(
                        int(scores[j, :, lm].argmax().cpu().numpy()))
                    topk_lm_class[lm].append(
                        list(scores[j, :, lm].sort(descending=True)[1]).index(
                            lab[j]))
            # Get landmark coordinates
            grid_x, grid_y = torch.meshgrid(torch.arange(maps.shape[2]),
                                            torch.arange(maps.shape[3]))
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

            map_sums = maps.sum(3).sum(2).detach()
            maps_x = grid_x * maps
            maps_y = grid_y * maps
            loc_x = maps_x.sum(3).sum(2) / map_sums
            loc_y = maps_y.sum(3).sum(2) / map_sums
    pbar.close()


if __name__=="__main__":
    main()