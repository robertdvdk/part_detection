"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
## Compositionality loss
# TODO reshuffle pixels to background instead of black background

# # CHANGE TO PER IMAGE INSTEAD OF PER BATCH
# random_landmark = np.random.randint(0, net.num_landmarks)
# random_map = maps[:, random_landmark:random_landmark+1,:,:]
# random_map_upsampled = torch.nn.functional.interpolate(random_map, size=(sample[0].shape[-2], sample[0].shape[-1]), mode='bilinear')
# map_argmax = torch.argmax(random_map, axis=0)
# mask = torch.where(map_argmax==random_landmark, 1, 0)
# # Permute dimensions: sample[0] is 12x3x256x256, random_map is 12x256x256
# # permute sample[0] to 3x12x256x256 so we can multiply them
# masked_imgs = torch.permute((torch.permute(sample[0], (1, 0, 2, 3))).to(device) * mask, (1, 0, 2, 3))

# with torch.no_grad():
#    masked_imgs = (sample[0].to(device)*random_map_upsampled)
#    _, _, _, comp_featuretensor, _ = net(masked_imgs)
#    masked_feature = (comp_featuretensor*random_map).mean(-1).mean(-1)

# # _, _, _, comp_featuretensor = net(masked_imgs)
# _, _, _, comp_featuretensor, _ = net(masked_imgs)
# masked_feature = (maps[:, random_landmark, :, :].unsqueeze(-1).permute(0,3,1,2) * comp_featuretensor).mean(-1).mean(-1)

# unmasked_feature = anchor[:, :, random_landmark]
# cos_sim_comp = torch.nn.functional.cosine_similarity(masked_feature.detach(), unmasked_feature, dim=-1)
# comp_loss = 1 - torch.mean(cos_sim_comp)
# loss_comp = comp_loss * l_comp
# Function definitions
def main():
    pass


if __name__ == "__main__":
    main()
