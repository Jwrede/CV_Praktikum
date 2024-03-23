import random 

import torch
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import conv2d
from sklearn.cluster import KMeans
from itertools import chain, combinations
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from visualize import *
from a2_net import SimpleMLP

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
    features = conv2d(greyscale_image_batch, masks, padding=5//2)
    texture_energy_map = conv2d(torch.abs(features), torch.ones(9,9,15,15), padding=15//2)
    # remove borders which are not part of the features
    return torch.cat([image_batch, texture_energy_map], dim=1)

def k_means(features, n_clusters=2, image_size=128):
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(features.view(12,-1).T)
    labels = kmeans.labels_.reshape(image_size,image_size)
    return labels

def binary_clustering(labels, seg, n_clusters=2, eval_points=10, image_size=128, pet_weighting=.4):
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
        accuracy = (pet_weighting*np.sum(sample_pet_mask) + (1-pet_weighting)*(eval_points - np.sum(sample_bg_mask))) / (eval_points*2)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_pet_clusters = s
    
    pet_mask = np.isin(labels, best_pet_clusters) + 1
    print(f"Best accuracy: {best_accuracy}, best pet clusters: {best_pet_clusters}")
    return pet_mask


def get_dataloader(filtered=True, image_size=128):
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
    if not filtered:
        return DataLoader(
            oxfordIIITPet,
            batch_size=4,
            num_workers=4,
        )
    with open("./data/oxford-iiit-pet/annotations/trainval.txt", "r") as f:
        trainval = [l.replace("\n", "").split(" ") for l in f.readlines()]
        df = pd.DataFrame(trainval, columns=['CLASS-ID', 'SPECIES', 'BREED', 'ID'])
    filtered_df = df.groupby('SPECIES').head(4)

    subset = Subset(oxfordIIITPet, filtered_df.index)
    data_loader = DataLoader(
        subset,
        batch_size=4,
        num_workers=4,
    )
    return data_loader

def a1():
    image_size=500
    data_loader = get_dataloader(filtered=True, image_size=image_size)

    for i, (rgb_image_batch, seg) in enumerate(data_loader):
        features = generate_features(rgb_image_batch, image_size)
        for j in range(4):
            n_clusters = 10
            labels = k_means(features[j], seg[j], n_clusters=n_clusters, image_size=image_size)
            if n_clusters > 2:
                labels = binary_clustering(labels, seg[j], n_clusters=n_clusters, image_size=image_size)
            save_images(rgb_image_batch, labels, seg, i, j, n_clusters=n_clusters, save=False)


def image_sampling(feature_batch, seg_batch, percentage=0.1, image_size=128):
    batch_size = seg_batch.shape[0]
    seg_batch = seg_batch.view(batch_size, image_size, image_size)
    seg_batch = seg_batch*255   
    
    flattened_image = feature_batch.view(12, -1).T
    flattened_seg = seg_batch.view(-1)
    
    all_indices = np.arange(len(flattened_image))
    indices = np.random.choice(all_indices, int(len(flattened_image)*percentage*batch_size), replace=False)
    sampled_image = flattened_image[indices]
    sampled_seg = flattened_seg[indices]
    sampled_image_test = flattened_image[np.setdiff1d(all_indices, indices)]
    sampled_seg_test = flattened_seg[np.setdiff1d(all_indices, indices)]
    
    sampled_seg = (sampled_seg > 1)
    sampled_seg_test = (sampled_seg_test > 1)
    return sampled_image, sampled_seg, sampled_image_test, sampled_seg_test, seg_batch

def visualize_results(image, true_seg, predicted_seg):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image.cpu().permute(1, 2, 0).numpy())
    axes[0].set_title('Original Image')
    axes[1].imshow(true_seg.cpu().squeeze(), cmap='gray')
    axes[1].set_title('True Segmentation')
    axes[2].imshow(predicted_seg, cmap='gray')
    axes[2].set_title('Predicted Segmentation')
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def predict_whole_image(image, model, image_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pixels = image.view(-1, 12).to(device)  # Reshape and move to the appropriate device

    model.eval()
    with torch.no_grad():
        outputs = model(pixels)
        _, predicted = torch.max(outputs, 1)
        predicted_seg = predicted.view(image_size, image_size).cpu().numpy()
    
    return predicted_seg


def train_model_for_image(image, seg, num_epochs=100, lr=0.01, image_size=128):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimpleMLP(2).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    num_pixels = image.shape[0]
    for epoch in range(num_epochs):
        loss_avg = 0
        for i in range(num_pixels):
            pixel = image[i].unsqueeze(0)  # Add batch dimension
            label = seg[i].unsqueeze(0).long()

            optimizer.zero_grad()
            output = model(pixel)
            loss = criterion(output, label)
            loss_avg += loss.item()
            loss.backward()
            optimizer.step()

        # if epoch % 10 == 0:  # Print loss every 10 epochs
        print(f'Epoch {epoch}, Loss: {loss_avg/num_pixels:.4f}')
    
    return model


def a2():
    image_size = 128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_loader = get_dataloader(filtered=True, image_size=image_size)
    num_epochs = 100

    for _, (rgb_image_batch, seg_batch) in enumerate(data_loader):
        for image, seg in zip(rgb_image_batch, seg_batch):
            image = image.view(1, 3, image_size, image_size)
            seg = seg.view(1, image_size, image_size)
            features = generate_features(image, image_size)
            sampled_image, sampled_seg, _, _, _ = image_sampling(features, seg, percentage=0.1, image_size=image_size)
            sampled_image, sampled_seg = sampled_image.to(device), sampled_seg.to(device)

            model = train_model_for_image(sampled_image, sampled_seg, 10, lr=0.01)
            predicted_seg_whole_image = predict_whole_image(features, model, image_size)
            print(torch.tensor(predicted_seg_whole_image).unique(return_counts=True))
            visualize_results(image[0], seg, predicted_seg_whole_image)


# def a2():
#     image_size=128
#     num_classes = 2
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     data_loader = get_dataloader(filtered=True, image_size=image_size)
#     model = SimpleMLP(num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     num_epochs = 100
#     writer = SummaryWriter()

#     image_nr = 0
#     for _, (rgb_image_batch, seg_batch) in enumerate(data_loader):
#         for image, seg in zip(rgb_image_batch, seg_batch):
#             image = image.view(1, 3, image_size, image_size)
#             seg = seg.view(1, image_size, image_size)
#             features = generate_features(image, image_size)
#             sampled_image, sampled_seg, _, _, _ = \
#                 (i.to(device) for i in image_sampling(features, seg, percentage=0.1, image_size=image_size))
#             for epoch in range(num_epochs):
#                 avg_loss = 0
#                 for pixel_image, pixel_seg in zip(sampled_image, sampled_seg):
#                     output = model(pixel_image)
                    
#                     # Calculate the loss
#                     loss = criterion(output, F.one_hot(pixel_seg.to(torch.int64), 2).float())
#                     avg_loss += loss.item()
                    
#                     optimizer.zero_grad()
#                     loss.backward()
#                     optimizer.step()
#                 # writer.add_scalar('Loss/train image' + str(image_nr), loss, image_nr)
#                 # plot whole image
#                 print(f"Average loss: {avg_loss/len(sampled_image):.4f}")
#                 result = torch.argmax(F.softmax(model(features.view(12,-1).T.to(device)), dim=1), dim=1)
#                 result = result.view(image_size, image_size)
#                 result = result.cpu().numpy()
#                 print(result)
#                 print(np.sum(result))
#                 plt.imshow(result)
#                 plt.show()
#             image_nr += 1

if __name__ == "__main__":
    a1()