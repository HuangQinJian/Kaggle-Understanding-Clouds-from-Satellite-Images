'''
@Description: 
@Author: HuangQinJian
@Date: 2019-10-04 15:47:50
@LastEditTime: 2019-10-06 18:50:07
@LastEditors: HuangQinJian
'''

import os
from glob import glob

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

TRAIN_PATH = 'data/test/'

train_df = pd.read_csv('data/submission.csv')

split_df = train_df["Image_Label"].str.split("_", n=1, expand=True)
# add new columns to train_df
train_df['Image'] = split_df[0]
train_df['Label'] = split_df[1]

print(train_df.head(10))
print(train_df.shape)


def get_mask_by_image_id(image, image_id, label):
    '''
    Function to visualize several segmentation maps.
    INPUT:
        image_id - filename of the image
    RETURNS:
        np_mask - numpy segmentation map
    '''
    im_df = train_df[train_df['Image'] == image_id.split('/')[-1]].fillna('-1')

    rle = im_df[im_df['Label'] == label]['EncodedPixels'].values[0]
    if rle != '-1':
        np_mask = rle_to_mask(rle, np.asarray(
            image).shape[1], np.asarray(image).shape[0])
        np_mask = np.clip(np_mask, 0, 1)
    else:
        # empty mask
        np_mask = np.zeros(
            (np.asarray(image).shape[0], np.asarray(image).shape[1]))

    return np_mask


def rle_to_mask(rle_string, width, height):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask

    Returns: 
    numpy.array: numpy array of the mask
    '''

    rows, cols = height, width

    if rle_string == -1:
        return np.zeros((height, width))
    else:
        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]
        rle_pairs = np.array(rle_numbers).reshape(-1, 2)
        img = np.zeros(rows*cols, dtype=np.uint8)
        for index, length in rle_pairs:
            index -= 1
            img[index:index+length] = 255
        img = img.reshape(cols, rows)
        img = img.T
        return img


def create_segmap(image, image_id):
    '''
    Helper function to create a segmentation map for an image by image filename
    '''

    # get masks for different classes
    fish_mask = get_mask_by_image_id(image, image_id, 'Fish')
    flower_mask = get_mask_by_image_id(image, image_id, 'Flower')
    gravel_mask = get_mask_by_image_id(image, image_id, 'Gravel')
    sugar_mask = get_mask_by_image_id(image, image_id, 'Sugar')

    # label numpy map with 4 classes
    segmap = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    segmap = np.where(fish_mask == 1, 1, segmap)
    segmap = np.where(flower_mask == 1, 2, segmap)
    segmap = np.where(gravel_mask == 1, 3, segmap)
    segmap = np.where(sugar_mask == 1, 4, segmap)

    # create a segmantation map
    segmap = SegmentationMapsOnImage(segmap, shape=image.shape, nb_classes=5)

    return segmap


def get_labels(image_id):
    ''' Function to get the labels for the image by name'''
    im_df = train_df[train_df['Image'] == image_id].fillna('-1')
    im_df = im_df[im_df['EncodedPixels'] != '-1'].groupby('Label').count()

    index = im_df.index
    all_labels = ['Fish', 'Flower', 'Gravel', 'Sugar']

    labels = ''

    for label in all_labels:
        if label in index:
            labels = labels + ' ' + label

    return labels


def draw_labels(image, np_mask, label):
    '''
    Function to add labels to the image.
    '''
    if np.sum(np_mask) > 0:
        x, y = 0, 0
        # very ugly code
        # I'm thinking on how to improve that
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if np_mask[i, j] > 0:
                    x = i
                    y = j
                    break
            if np_mask[i, j] > 0:
                x = i
                y = j
                break

        image = imgaug.imgaug.draw_text(
            image, x, y, label, color=(255, 255, 255), size=50)
    return image


def draw_segmentation_maps(image, image_id):
    '''
    Helper function to draw segmantation maps and text.
    '''

    # get masks for different classes
    fish_mask = get_mask_by_image_id(image, image_id, 'Fish')
    flower_mask = get_mask_by_image_id(image, image_id, 'Flower')
    gravel_mask = get_mask_by_image_id(image, image_id, 'Gravel')
    sugar_mask = get_mask_by_image_id(image, image_id, 'Sugar')

    # label numpy map with 4 classes
    segmap = create_segmap(image, image_id)

    # draw the map on image
    image = np.asarray(segmap.draw_on_image(image)
                       ).reshape(image.shape)

    image = draw_labels(image, fish_mask, 'Fish')
    image = draw_labels(image, flower_mask, 'Flower')
    image = draw_labels(image, gravel_mask, 'Gravel')
    image = draw_labels(image, sugar_mask, 'Sugar')

    return image


def plot_training_images_and_masks(width=2, height=3):
    """
    Function to plot grid with several examples of cloud images from train set.
    INPUT:
        width - number of images per row
        height - number of rows

    OUTPUT: None
    """

    # get a list of images from training set
    images = sorted(os.listdir(TRAIN_PATH))
    images = [os.path.join(TRAIN_PATH, path) for path in images]

    fig, axs = plt.subplots(height, width, figsize=(16, 16))

    # create a list of random indices
    rnd_indices = rnd_indices = [np.random.choice(
        range(0, len(images))) for i in range(height * width)]

    for im in range(0, height * width):
        image_id = images[rnd_indices[im]]
        image = np.asarray(Image.open(image_id))
        image = cv2.resize(image, dsize=(525, 350),
                           interpolation=cv2.INTER_LINEAR)
        image = np.asarray(image)
        image = draw_segmentation_maps(image, image_id)

        i = im // width
        j = im % width

        # plot the image
        axs[i, j].imshow(image)  # plot the data
        axs[i, j].axis('off')
        axs[i, j].set_title(get_labels(images[rnd_indices[im]].split('/')[-1]))

    # set suptitle
    plt.suptitle('Sample images from the test set')
    plt.show()


plot_training_images_and_masks()
