import os

import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image

def display_images_and_masks(dataset, indexes):
    """Display images and segmentation masks next to each other.
    """
    # Display a maximum of 2 sets of (image, mask) pairs per row.
    nrows = (len(indexes) + 1) // 2
    # 3 units height per row.
    fig = plt.figure(figsize=(10, 3 * nrows))
    for i in range(len(indexes)):
        image, mask = dataset[i][0], dataset[i][1]
        fig.add_subplot(nrows, 4, i*2+1)
        plt.imshow(image)
        plt.axis("off")
        
        fig.add_subplot(nrows, 4, i*2+2)
        plt.imshow(mask)
        plt.axis("off")
    plt.show()

def get_concat_h(*args):
    dst = Image.new('RGB', (sum(img.width for img in args), args[0].height))
    for i, im in enumerate(args):
        dst.paste(im, (i*im.width, 0))
    return dst

def save_images(rgb_image_batch, labels, seg, i, j, n_clusters=2, show=True, save=True):
    feature_image = torchvision.transforms.ToPILImage()(torch.tensor(labels).float() / n_clusters)
    original_image = torchvision.transforms.ToPILImage()(rgb_image_batch[j])
    seg_image = torchvision.transforms.ToPILImage()(seg[j]*255/n_clusters)
    concat_image = get_concat_h(original_image, feature_image, seg_image)
    os.makedirs(f"./output/A1c/{n_clusters}", exist_ok=True)
    if show:
        concat_image.show()
    if save:
        concat_image.save(f"./output/A1c/{n_clusters}/{j+i*4}.png")