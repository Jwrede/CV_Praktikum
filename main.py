from visualize import *
import random 

import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.nn.functional import conv2d
from torchvision.transforms.functional import crop
from sklearn.cluster import KMeans
from itertools import chain, combinations

def filter_df(df):
    """ Get only 4 images for each species"""
    return df.groupby('SPECIES').head(4)

def get_basis_vectors():
    L5 = torch.tensor([1, 4, 6, 4, 1]).reshape(1, 5)
    E5 = torch.tensor([-1, -2, 0, 2, 1]).reshape(1, 5)
    S5 = torch.tensor([-1, 0, 2, 0, -1]).reshape(1, 5)
    R5 = torch.tensor([1, -4, 6, -4, 1]).reshape(1, 5)
    return L5, E5, S5, R5

def get_nine_masks():
    L5, E5, S5, R5 = get_basis_vectors()
    E5E5 = torch.matmul(E5.T, E5)
    S5S5 = torch.matmul(S5.T, S5)
    R5R5 = torch.matmul(R5.T, R5)
    L5E5E5L5 = (torch.matmul(L5.T, E5) + torch.matmul(E5.T, L5)) / 2
    L5S5S5L5 = (torch.matmul(L5.T, S5) + torch.matmul(S5.T, L5)) / 2
    L5R5R5L5 = (torch.matmul(L5.T, R5) + torch.matmul(R5.T, L5)) / 2
    E5S5S5E5 = (torch.matmul(E5.T, S5) + torch.matmul(S5.T, E5)) / 2
    E5R5R5E5 = (torch.matmul(E5.T, R5) + torch.matmul(R5.T, E5)) / 2
    S5R5R5S5 = (torch.matmul(S5.T, R5) + torch.matmul(R5.T, S5)) / 2
    return torch.stack([E5E5, S5S5, R5R5, L5E5E5L5, L5S5S5L5, L5R5R5L5, E5S5S5E5, E5R5R5E5, S5R5R5S5])

def generate_features(image_batch, image_size):
    greyscale_image_batch = torchvision.transforms.Grayscale(num_output_channels=1)(image_batch)
    masks = get_nine_masks().view(9,1,5,5)
    features = conv2d(greyscale_image_batch, masks, padding=0)
    # remove borders which are not part of the features
    image_batch = crop(image_batch, 2, 2, image_size-4, image_size-4)
    return torch.cat([image_batch, features], dim=1)

def k_means(features, seg, n_clusters=2, image_size=128):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(features.view(12,-1).T)
    labels = kmeans.labels_.reshape(image_size-4,image_size-4)
    return labels

def binary_clustering(labels, seg, n_clusters=2, eval_points=10, image_size=128, pet_weighting=.8):
    """ Evaluate k-means by comparing the segmentation masks with the labels.
    Compare `eval_points` points in the segmentation mask with the labels.
    group the clusters into two groups, one for pet and one for background.
    Use a powerset to get all possible combinations of the clusters.
    Then compare the segmentation mask with the labels.
    pet weighting is the weight of the pet clusters in the accuracy calculation since it's harder
    to classify the pets in most pictures.
    """
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1,len(s)))

    seg = seg.view(image_size, image_size)
    seg = crop(seg, 2, 2, image_size-4, image_size-4)
    seg = seg.numpy()*255

    indices_x_pet, indices_y_pet = (seg > 1).nonzero()
    random_sample_pet = random.sample(range(len(indices_y_pet)), eval_points)
    indices_x_pet, indices_y_pet = indices_x_pet[random_sample_pet], indices_y_pet[random_sample_pet]

    indices_x_bg, indices_y_bg = (seg > 1).nonzero()
    random_sample_bg = random.sample(range(len(indices_y_bg)), eval_points)
    indices_x_bg, indices_y_bg = indices_x_bg[random_sample_bg], indices_y_bg[random_sample_bg]

    pet_clusters = list(powerset(range(n_clusters)))
    best_accuracy = -1
    best_pet_clusters = None
    for s in pet_clusters:
        sample_pet_mask = np.isin(labels[indices_x_pet, indices_y_pet], s)
        sample_bg_mask = np.isin(labels[indices_x_bg, indices_y_bg], s)
        accuracy = (pet_weighting*np.sum(sample_pet_mask) + (1-pet_weighting)*(eval_points - np.sum(sample_bg_mask))) / eval_points*2
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pet_clusters = s
    
    pet_mask = np.isin(labels, best_pet_clusters) + 1
    print(f"Best accuracy: {best_accuracy}, best pet clusters: {best_pet_clusters}")
    return pet_mask


def main():
    image_size = 128
    oxfordIIITPet = torchvision.datasets.OxfordIIITPet(
        "./data",
        download=True,
        target_types = "segmentation",
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor()
        ]),
        target_transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor()
        ])
    )
    with open("./data/oxford-iiit-pet/annotations/trainval.txt", "r") as f:
        trainval = [l.replace("\n", "").split(" ") for l in f.readlines()]
        df = pd.DataFrame(trainval, columns=['CLASS-ID', 'SPECIES', 'BREED', 'ID'])
    filtered_df = filter_df(df)

    subset = Subset(oxfordIIITPet, filtered_df.index)
    data_loader = DataLoader(
        subset,
        batch_size=4,
        num_workers=4,
    )

    for i, (rgb_image_batch, seg) in enumerate(data_loader):
        features = generate_features(rgb_image_batch, image_size)
        for j in range(4):
            n_clusters = 10            
            labels = k_means(features[j], seg[j], n_clusters=n_clusters)
            if n_clusters > 2:
                labels = binary_clustering(labels, seg[j], n_clusters=n_clusters, image_size=image_size)
            save_images(rgb_image_batch, labels, seg, i, j, n_clusters=n_clusters, save=False)


if __name__ == "__main__":
    main()