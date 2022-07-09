import pandas as pd
import numpy as np
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from mpl_toolkits.axes_grid1 import make_axes_locatable


print(torch.cuda.is_available())


data_path = "/home/diego/Downloads/Datasets/happyWhale"


class WhaleDataset(torch.utils.data.Dataset):
    """Whale dataset."""
    def __init__(self,data_path,mode='train',height=256,minimum_images=3, alt_data_path = None):
        """
        Args:
            data_path (string): path to the dataset
            mode (string): 'train' or 'val'
        """
        self.data_path = data_path
        self.alt_data_path = alt_data_path
        train_data = pd.read_csv(os.path.join(data_path,'train.csv'))
        unique_labels, unique_label_counts = np.unique(train_data['Id'],return_counts=True)


        # Remove classes with less than 3 photos
        unique_labels = unique_labels[unique_label_counts>=minimum_images]

        # Remove new_whale
        unique_labels = unique_labels[1:]

        # Create vector of labels and set ids (1 for train, 2 for test)
        self.unique_labels = list(unique_labels)
        labels = []
        label_ids = []
        setid = []
        names = []
        unique_labels_seen = np.zeros(len(self.unique_labels))
        for i in range(len(train_data)):
            if train_data['Id'][i] in self.unique_labels:
                labels.append(self.unique_labels.index(train_data['Id'][i]))
                label_ids.append(train_data['Id'][i])
                names.append(train_data['Image'][i])
                if unique_labels_seen[labels[-1]] == 0:
                    setid.append(2)
                else:
                    setid.append(1)
                unique_labels_seen[labels[-1]] += 1
        self.mode = mode
        if mode == 'train':
            self.labels = np.array(labels)[np.array(setid)==1]
            self.label_ids = np.array(label_ids)[np.array(setid)==1]
            #self.labels = np.vstack((self.labels*2,self.labels*2+1)).T.reshape(-1)
            self.names = np.array(names)[np.array(setid)==1]
        if mode == 'val':
            self.labels = np.array(labels)[np.array(setid)==2]
            self.label_ids = np.array(label_ids)[np.array(setid)==2]
            self.names = np.array(names)[np.array(setid)==2]
        if mode == 'no_set':
            self.labels = np.array(labels)
            self.label_ids = np.array(label_ids)
            self.names = np.array(names)
        self.height = height

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #if self.mode=='train':
        #    do_flip = idx%2
        #    im_idx = int(np.floor(idx/2))
        #else:
        #    do_flip = 0
        #    im_idx = idx
        if self.alt_data_path is not None and os.path.isfile(os.path.join(self.alt_data_path,self.names[idx])):
            im = cv2.imread(os.path.join(self.alt_data_path,self.names[idx]))
            im = np.flip(im,2)
        else:
            im = cv2.imread(os.path.join(self.data_path,'train',self.names[idx]))
        im = cv2.resize(im,(self.height*2,self.height))
        label = self.labels[idx]
        #if do_flip:
        #    im = cv2.flip(im,1)

        if len(im.shape) == 2:
          im = np.stack((im,)*3, axis=-1)

        im = np.float32(np.transpose(im,axes=(2,0,1)))/255

        return (im,label)

class WhaleTripletDataset(torch.utils.data.Dataset):
    """Whale dataset."""
    def __init__(self,orig_dataset,height_list=[256,256,256]):
        """
        Args:
            orig_dataset (Dataset): dataset
        """
        self.orig_dataset = orig_dataset
        self.height_list = height_list
    def __len__(self):
        return len(self.orig_dataset)

    def __getitem__(self, idx):
        self.orig_dataset.height = self.height_list[0]
        im,lab = self.orig_dataset[idx]
        opts = np.where(self.orig_dataset.labels == lab)[0]
        positive_idx = opts[np.random.randint(len(opts))]
        opts = np.where(self.orig_dataset.labels != lab)[0]
        negative_idx = opts[np.random.randint(len(opts))]
        self.orig_dataset.height = self.height_list[1]
        im_pos,_ = self.orig_dataset[positive_idx]
        self.orig_dataset.height = self.height_list[2]
        im_neg,lab_neg = self.orig_dataset[negative_idx]
        return (im,im_pos,im_neg,lab,lab_neg)

class Net(torch.nn.Module):
  def __init__(self,init_model):
    super().__init__()
    self.conv1 = init_model.conv1
    self.bn1 = init_model.bn1
    self.relu = init_model.relu
    self.maxpool = init_model.maxpool
    self.layer1 = init_model.layer1
    self.layer2 = init_model.layer2
    self.layer3 = init_model.layer3
    self.layer4 = init_model.layer4
    self.layer3[0].downsample[0].stride = (1,1)
    self.layer3[0].conv1.stride = (1,1)
    self.layer4[0].downsample[0].stride = (1,1)
    self.layer4[0].conv1.stride = (1,1)
    self.finalpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc = torch.nn.Linear(512,300,bias=False)
    self.fc_class = torch.nn.Linear(300,2000,bias=False)
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.finalpool(x).squeeze(-1).squeeze(-1)
    x = self.fc(x)
    y = self.fc_class(x)
    return x,x,y,x

class LandmarkNet(torch.nn.Module):
  def __init__(self,init_model,num_landmarks=8):
    super().__init__()
    self.conv1 = init_model.conv1
    self.bn1 = init_model.bn1
    self.relu = init_model.relu
    self.maxpool = init_model.maxpool
    self.layer1 = init_model.layer1
    self.layer2 = init_model.layer2
    self.layer3 = init_model.layer3
    self.layer4 = init_model.layer4
    self.layer3[0].downsample[0].stride = (1,1)
    self.layer3[0].conv1.stride = (1,1)
    self.layer4[0].downsample[0].stride = (1,1)
    self.layer4[0].conv1.stride = (1,1)
    self.fc = torch.nn.Conv2d(512,300,1,bias=False)
    self.pool = torch.nn.AdaptiveAvgPool2d(1)
    #self.fc_global = torch.nn.Conv2d(512,512,1)
    self.fc_landmarks = torch.nn.Conv2d(512,num_landmarks+1,1,bias=False)
    self.fc_class = torch.nn.Linear(300,2000,bias=False)
    #self.landmark_mask = torch.nn.Parameter(torch.zeros(1,300,10-1))
    #torch.nn.init.normal_(self.landmark_mask,std=1)
    #self.landmark_proj = torch.nn.Parameter(torch.Tensor(300,300,num_landmarks))
    #torch.nn.init.kaiming_uniform_(self.landmark_proj)
    #self.landmark_proj = torch.nn.Parameter(torch.eye(300,300).unsqueeze(-1).repeat(1,1,10-1))
    #self.dropout = torch.nn.Dropout2d(0.5) 
    self.softmax=torch.nn.Softmax2d()
    self.avg_dist_pos = torch.nn.Parameter(torch.zeros([num_landmarks]),requires_grad=False)
    self.avg_dist_neg = torch.nn.Parameter(torch.zeros([num_landmarks]),requires_grad=False)
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    #maps = self.fc_landmarks(x*self.fc_global(self.pool(x)))
    maps = self.fc_landmarks(x)
    maps = self.softmax(maps)
    #maps_dropped = self.dropout(maps)
    #maps_v = maps.reshape([maps.shape[0],maps.shape[1],maps.shape[2]*maps.shape[3]])
    #attn = maps_v.softmax(1)
    x = self.fc(x)
    #x = torch.nn.functional.normalize(x)
    #x_v = x.reshape([x.shape[0],x.shape[1],x.shape[2]*x.shape[3]])
    #x = (attn.unsqueeze(-1).permute(0,3,2,1)*x_v.unsqueeze(-1)).mean(3).mean(2)
    #x = (attn.unsqueeze(-1).permute(0,3,2,1)*x_v.unsqueeze(-1)).mean(3).mean(2)
    feature_tensor = x
    x = (maps[:,0:-1,:,:].unsqueeze(-1).permute(0,4,2,3,1)*x.unsqueeze(-1)).mean(2).mean(2)
    #x = x * (self.landmark_mask).sigmoid()
    #new_x = []
    #for i in range(x.shape[-1]):
    #    new_x.append(torch.matmul(x[:,:,i],self.landmark_proj[:,:,i]).unsqueeze(-1))
    #x = torch.cat(new_x,2)
    #x = x.sum(2)
    #x = torch.nn.functional.normalize(x)
    y = self.fc_class(x.permute(0,2,1)).permute(0,2,1)
    return x,maps,y,feature_tensor





dataset_train = WhaleDataset(data_path,mode='train')
dataset_val = WhaleDataset(data_path,mode='val')
dataset_full = WhaleDataset(data_path,mode='no_set',minimum_images=0,alt_data_path='Teds_OSM')
dataset_train_triplet = WhaleTripletDataset(dataset_train)

batch_size = 12
train_loader = torch.utils.data.DataLoader(dataset=dataset_train_triplet, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

number_epochs = 40
model_name = 'landmarks_10_nodrop_4.pt'
model_name_init = 'landmarks_10_nodrop_4.pt'
warm_start = True
do_only_test = True


do_baseline = False
num_landmarks = 10
triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
classif_loss = torch.nn.CrossEntropyLoss()
basenet = torchvision.models.resnet18(pretrained=True)
if do_baseline:
    net = Net(basenet)
else:
    net = LandmarkNet(basenet,num_landmarks)
if warm_start:
  net.load_state_dict(torch.load(model_name_init),strict=False)
net.to(device)

optimizer = torch.optim.Adam([{'params': list(net.parameters())[0:-1], 'lr': 1e-4},
                              {'params': list(net.parameters())[-1:], 'lr': 1e-2}])

if do_only_test:
    number_epochs = 1

val_accs = []

test_batch = []
test_batch_labels = []
for id in test_list:
    num = list(dataset_full.names).index(id)
    test_batch.append(torch.Tensor(dataset_full[num][0]).unsqueeze(0))
    test_batch_labels.append(dataset_full.unique_labels[dataset_full[num][1]])


for epoch in range(number_epochs):
    if not do_only_test:
        # Training
        net.train()
        all_training_vectors = []
        all_training_labels = []
        pbar = tqdm(total=len(train_loader),position=0, leave=True)
        iter_loader = iter(train_loader)
        for i in range(len(train_loader)):
            if i%100 == -1:
                train_loader.dataset.height_list[0] = np.random.randint(200,300)
                train_loader.dataset.height_list[1] = np.random.randint(200,300)
                train_loader.dataset.height_list[2] = np.random.randint(200,300)
                iter_loader = iter(train_loader)
            sample = next(iter_loader)
            with torch.no_grad():
                anchor,_,_,_ = net(sample[0].to(device))
                all_training_vectors.append(anchor.cpu())
                all_training_labels.append(sample[3])

            # Flip the positive and the anchor
            do_class = 1
            if np.random.rand()>0.5:
                sample[0] = sample[0].flip(-1)
                sample[1] = sample[1].flip(-1)
                do_class = 0
            # Flip the negative
            if np.random.rand()>0.5:
                sample[2] = sample[2].flip(-1)
            
            # Substitute the negative with the flipped positive or anchor
            if np.random.rand()>0.9:
                if np.random.rand()>0.5:
                    sample[2] = sample[0].flip(-1)
                else:
                    sample[2] = sample[1].flip(-1)
            
            angle = np.random.randn()*0.1
            scale = np.random.rand()*0.2 + 0.9
            sample[0] = torchvision.transforms.functional.affine(sample[0],angle=angle*180/np.pi,translate=[0,0],scale=scale,shear=0)
            angle = np.random.randn()*0.1
            scale = np.random.rand()*0.2 + 0.9
            sample[1] = torchvision.transforms.functional.affine(sample[1],angle=angle*180/np.pi,translate=[0,0],scale=scale,shear=0)
            angle = np.random.randn()*0.1
            scale = np.random.rand()*0.2 + 0.9
            sample[2] = torchvision.transforms.functional.affine(sample[2],angle=angle*180/np.pi,translate=[0,0],scale=scale,shear=0)
            
            anchor,maps,scores_anchor,feature_tensor = net(sample[0].to(device))
            positive,_,scores_pos,_ = net(sample[1].to(device))
            negative,_,_,_ = net(sample[2].to(device))

            if not do_baseline:
                net.avg_dist_pos.data = net.avg_dist_pos.data*0.95 +  ((anchor.detach()-positive.detach())**2).mean(0).sum(0).sqrt()*0.05
                net.avg_dist_neg.data = net.avg_dist_neg.data*0.95 +  ((anchor.detach()-negative.detach())**2).mean(0).sum(0).sqrt()*0.05
            

            
            #for lm in range(scores_anchor.shape[-1]):
            #    loss_class += ( classif_loss(scores_anchor[:,:,lm],sample[3].to(device)) + classif_loss(scores_pos[:,:,lm],sample[3].to(device)) )/scores_anchor.shape[-1]
            
            #loss = triplet_loss((dropout_mask*anchor).mean(2)*d,(dropout_mask*positive).mean(2)*d,(dropout_mask*negative).mean(2)*d)
            #loss = triplet_loss((anchor).mean(2),(positive).mean(2),(negative).mean(2))/2 #+ triplet_loss((anchor[:,:,0:-1]),(positive[:,:,0:-1]),(negative[:,:,0:-1]))/2

            if do_baseline:
                loss = triplet_loss(anchor,positive,negative)
                loss_class = classif_loss(scores_anchor,sample[3].to(device))/2 + classif_loss(scores_pos,sample[3].to(device))/2
                total_loss = loss + 10*loss_class*do_class
                loss_conc = total_loss.detach()*0
                loss_max = total_loss.detach()*0
                loss_mean = total_loss.detach()*0
            else:
                loss = 0#triplet_loss((anchor).mean(2),(positive).mean(2),(negative).mean(2))
                loss_class = classif_loss(scores_anchor.mean(-1),sample[3].to(device))/2 + classif_loss(scores_pos.mean(-1),sample[3].to(device))/2
                for lm in range(anchor.shape[2]):
                    loss += triplet_loss((anchor[:,:,lm]),(positive[:,:,lm]),(negative[:,:,lm]))/((anchor.shape[2]-1))
                    #loss_class += (classif_loss(scores_anchor[:,:,lm],sample[3].to(device))/2 + classif_loss(scores_pos[:,:,lm],sample[3].to(device))/2)/((anchor.shape[2]-1))
                
                for drops in range(0):
                #dropout_mask = (torch.rand(1,1,scores_anchor.shape[-1])>np.random.rand()*0.5).float().to(device)
                    dropout_mask = (torch.rand(1,1,scores_anchor.shape[-1])>0.5).float().to(device)
                    d = 1/(dropout_mask.mean()+1e-6)
                    loss_class += classif_loss((dropout_mask*scores_anchor).mean(-1)*d,sample[3].to(device))/10
                    loss_class += classif_loss((dropout_mask*scores_pos).mean(-1)*d,sample[3].to(device))/10
                # Get landmark coordinates
                grid_x, grid_y = torch.meshgrid( torch.arange(maps.shape[2]), torch.arange(maps.shape[3]))
                grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
                grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

                map_sums = maps.sum(3).sum(2).detach()
                maps_x = grid_x * maps
                maps_y = grid_y * maps 
                loc_x = maps_x.sum(3).sum(2) / map_sums
                loc_y = maps_y.sum(3).sum(2) / map_sums

                # Apply model to transformed anchor
                #angle = (np.random.rand()-0.5)*0.5
                #scale = np.random.rand()*0.5+0.5
                #theta = torch.Tensor([[np.cos(angle), -1.0*np.sin(angle)],
                #                    [np.sin(angle), np.cos(angle)]]).to(device)
                #rot = torch.matmul(theta.T,torch.cat((loc_x.unsqueeze(1)-maps.shape[2]/2,loc_y.unsqueeze(1)-maps.shape[3]/2),1)).detach()
                #tloc_x = scale*rot[:,0,:]+maps.shape[2]/2
                #tloc_y = scale*rot[:,1,:]+maps.shape[3]/2
                #_,maps_resized = net(torchvision.transforms.functional.affine(sample[0],angle=angle*180/np.pi,translate=[0,0],scale=scale,shear=0).to(device))

                # Get landmark coordinates of resized version
                #grid_x_resized, grid_y_resized = torch.meshgrid( torch.arange(maps_resized.shape[2]), torch.arange(maps_resized.shape[3]))
                #grid_x_resized = grid_x_resized.unsqueeze(0).unsqueeze(0).to(device)
                #grid_y_resized = grid_y_resized.unsqueeze(0).unsqueeze(0).to(device)

                #map_sums_resized = maps_resized.sum(3).sum(2)
                #maps_x_resized = grid_x_resized * maps_resized
                #maps_y_resized = grid_y_resized * maps_resized 
                #loc_x_resized = maps_x_resized.sum(3).sum(2) / maps_resized.sum(3).sum(2)
                #loc_y_resized = maps_y_resized.sum(3).sum(2) / maps_resized.sum(3).sum(2)

                # Scale equivariance loss
                #scale_loss_x = (tloc_x - loc_x_resized/maps_resized.shape[-1]*maps.shape[-1])**2
                #scale_loss_y = (tloc_y - loc_y_resized/maps_resized.shape[-1]*maps.shape[-1])**2
                #loss_scale = scale_loss_x[:,0:-1].mean() + scale_loss_y[:,0:-1].mean()

                # Concentration loss
                loss_conc_x = (loc_x.unsqueeze(-1).unsqueeze(-1) - grid_x)**2
                loss_conc_y = (loc_y.unsqueeze(-1).unsqueeze(-1) - grid_y)**2
                loss_conc = ((loss_conc_x+loss_conc_y))*maps
                loss_conc = loss_conc[:,0:-1,:,:].mean()

                #loc_dist = torch.cdist(torch.cat((loc_x.unsqueeze(-1),loc_y.unsqueeze(-1)),2),torch.cat((loc_x.unsqueeze(-1),loc_y.unsqueeze(-1)),2))
                #loss_dist = (-loc_dist*1).exp()
                #loss_dist = (1-torch.eye(loss_dist.shape[-1]).to(device))*loss_dist
                #loss_dist = loss_dist.mean()

                #mask_loss = torch.matmul((1*net.landmark_mask[0,:,:]).T.sigmoid(),(1*net.landmark_mask[0,:,:]).sigmoid().log())
                #mask_loss = (1-torch.eye(mask_loss.shape[0]).to(device))*mask_loss
                #mask_loss = mask_loss.mean()

                loss_max=maps.max(-1)[0].max(-1)[0].mean()
                loss_max=1-loss_max

                loss_mean=maps[:,0:-1,:,:].mean()

                #choose_landmark = np.random.randint(maps.shape[1]-1)
                #upscaled_maps = torchvision.transforms.functional.resize(maps[:,choose_landmark:choose_landmark+1,:,:].detach(),256)
                #masked_input = upscaled_maps * sample[0].to(device)
                #_,_,_,x = net(masked_input)
                #masked_anchor = (maps[:,0:-1,:,:].detach().unsqueeze(-1).permute(0,4,2,3,1)*x.unsqueeze(-1)).mean(2).mean(2)
                #masked_diff = (torch.nn.functional.normalize(anchor[:,:,choose_landmark]) - torch.nn.functional.normalize(masked_anchor[:,:,choose_landmark]))**2
                #loss_masked_diff = masked_diff.sum(-1).mean()
                
                #feature_magnitudes = (feature_tensor**2).mean(1).sqrt().clamp_max(1)
                #selected_class = scores_anchor.softmax(1).mean(-1).sort(1,descending=True)[1][:,0]
                #prototype_scores = torch.cat([scores_anchor.softmax(1)[i:i+1,selected_class[i],:] for i in range(len(selected_class))])



                total_loss = loss + 1*loss_conc + 0*loss_mean + 1*loss_max + 1*loss_class*do_class #+ 1*loss_masked_diff

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            

            

            if epoch==0 and i==0:
                running_loss = loss.item()
                running_loss_conc = loss_conc.item()
                running_loss_mean = loss_mean.item()
                running_loss_max = loss_max.item()
                running_loss_class = loss_class.item()
                #running_loss_masked_diff = loss_masked_diff.item()
            else:
                running_loss = 0.99*running_loss + 0.01*loss.item()
                running_loss_conc = 0.99*running_loss_conc + 0.01*loss_conc.item()
                running_loss_mean = 0.99*running_loss_mean + 0.01*loss_mean.item()
                running_loss_max = 0.99*running_loss_max + 0.01*loss_max.item()
                running_loss_class = 0.99*running_loss_class + 0.01*loss_class.item()
                #running_loss_masked_diff = 0.99*running_loss_masked_diff + 0.01*loss_masked_diff.item()
            pbar.set_description("Training loss: %f, Conc: %f, Mean: %f, Max: %f, Class: %f" % (running_loss,running_loss_conc,running_loss_mean,running_loss_max,running_loss_class) )
            pbar.update()
            #if i == 15:
            #    break
        pbar.close()
        
        torch.save(net.cpu().state_dict(), model_name)
    # Validation
    net.eval()
    net.to(device)
    #training_feats = torch.cat(all_training_vectors).squeeze().to(device)
    #training_labs = torch.cat(all_training_labels).squeeze().to(device)
    pbar = tqdm(val_loader,position=0, leave=True)
    #topk = []
    #topk_lm = None
    top_class = []
    names = []
    topk_class = []
    diff_to_second = []
    topk_lm_class = None
    class_lm = None
    all_scores = []
    all_labels = []



    for i, sample in enumerate(pbar):
        feat,maps,scores,feature_tensor = net(sample[0].to(device))
        scores = scores.detach().cpu()
        all_scores.append(scores)
        lab = sample[1]
        all_labels.append(lab)
        #training_labs_expanded = training_labs.unsqueeze(0).expand(feat.shape[0],len(training_labs))
        #dist = torch.cdist(feat.mean(2).squeeze(),training_feats.mean(2))
        #sorted_labels = torch.gather(training_labs_expanded,1,dist.sort()[1])
        
        if do_baseline:
            for j in range(scores.shape[0]):
                #topk.append(list(sorted_labels[j,:]).index(lab[j])) 
                sorted_scores, sorted_indeces = scores[j,:].softmax(0).sort(descending=True)
                topk_class.append(list(sorted_indeces).index(lab[j]))
                top_class.append(sorted_indeces[0])
                diff_to_second.append(float(sorted_scores[0]-sorted_scores[1]))
        else:
            for j in range(scores.shape[0]):
                #topk.append(list(sorted_labels[j,:]).index(lab[j])) 
                sorted_scores, sorted_indeces = scores[j,:,:].mean(-1).softmax(0).sort(descending=True)
                topk_class.append(list(sorted_indeces).index(lab[j]))
                top_class.append(sorted_indeces[0])
                diff_to_second.append(float(sorted_scores[0]-sorted_scores[1]))
            if topk_lm_class is None:
                #topk_lm = []
                topk_lm_class = []
                class_lm = []
                for lm in range(feat.shape[2]):
                    #topk_lm.append([])
                    topk_lm_class.append([])
                    class_lm.append([])
            for lm in range(feat.shape[2]):
                #dist = torch.cdist(feat.squeeze()[:,:,lm],training_feats[:,:,lm])
                #sorted_labels = torch.gather(training_labs_expanded,1,dist.sort()[1])
                for j in range(scores.shape[0]):
                    #topk_lm[lm].append(list(sorted_labels[j,:]).index(lab[j])) 
                    class_lm[lm].append(int(scores[j,:,lm].argmax().cpu().numpy()))
                    topk_lm_class[lm].append(list(scores[j,:,lm].sort(descending=True)[1]).index(lab[j]))
            # Get landmark coordinates
            grid_x, grid_y = torch.meshgrid( torch.arange(maps.shape[2]), torch.arange(maps.shape[3]))
            grid_x = grid_x.unsqueeze(0).unsqueeze(0).to(device)
            grid_y = grid_y.unsqueeze(0).unsqueeze(0).to(device)

            map_sums = maps.sum(3).sum(2).detach()
            maps_x = grid_x * maps
            maps_y = grid_y * maps 
            loc_x = maps_x.sum(3).sum(2) / map_sums
            loc_y = maps_y.sum(3).sum(2) / map_sums

    pbar.close()
    