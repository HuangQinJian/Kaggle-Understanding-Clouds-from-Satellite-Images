'''
@Description: 
@Author: HuangQinJian
@Date: 2019-10-04 20:46:01
@LastEditTime: 2019-10-07 12:15:48
@LastEditors: HuangQinJian
'''

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import albumentations as albu
# import segmentation_models_pytorch as smp
import segmentation_models_pytorch_fpn_modify as smp
from clouddataset import CloudDataset
from tqdm import tqdm

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax'
MODEL_NAME = 'FPN'

CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']

WIDTH = 525
HEIGHT = 350

WIDTH_TRAIN = 1280
HEIGHT_TRAIN = 640

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

class_params = {0: (0.65, 10000), 1: (0.75, 10000),
                2: (0.65, 10000), 3: (0.75, 10000)}


def get_test_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.Resize(HEIGHT_TRAIN, WIDTH_TRAIN)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)


def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def resize_it(x):
    if x.shape != (HEIGHT, WIDTH):
        x = cv2.resize(x, dsize=(WIDTH, HEIGHT),
                       interpolation=cv2.INTER_LINEAR)
    return x


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    # print(threshold, min_size)
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((HEIGHT, WIDTH), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def tta_forward(dataloader, model):
    model.eval()
    with torch.no_grad():
        encoded_pixels = []
        image_id = 0
        for data in tqdm(dataloader):
            image = data[0].float().to(DEVICE)  # 复制image到model所在device上

            predict_1 = model(image)

            predict_2 = model(torch.flip(image, [-1]))
            predict_2 = torch.flip(predict_2, [-1])

            predict_3 = model(torch.flip(image, [-2]))
            predict_3 = torch.flip(predict_3, [-2])

            predict_4 = model(torch.flip(image, [-1, -2]))
            predict_4 = torch.flip(predict_4, [-1, -2])

            predict_list = (predict_1 + predict_2 + predict_3 + predict_4)/4

            # predict_list = torch.argmax(
            #     predict_list.cpu(), 1).byte().numpy()  # n x h x w

            # print(predict_list.shape)

            for per_img in predict_list:
                # print(per_img.shape)
                for probability in per_img:
                    probability = probability.cpu().detach().numpy()
                    probability = resize_it(probability)
                    # print(probability.shape)
                    def sigmoid(x): return 1 / (1 + np.exp(-x))
                    # predict, num_predict = post_process(
                    #     sigmoid(probability), threshold, min_size)
                    predict, num_predict = post_process(
                        sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                    if num_predict == 0:
                        encoded_pixels.append('')
                    else:
                        r = mask2rle(predict)
                        encoded_pixels.append(r)
                    image_id += 1

    return encoded_pixels


def get_arguments():
    parser = argparse.ArgumentParser(description="data_prepare")
    parser.add_argument("--weights_path", type=str, required=True,
                        help="the path to model weight")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="batch_size")
    return parser.parse_args()


def main():
    args = get_arguments()
    weights_path = args.weights_path
    batch_size = args.batch_size

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    sub = pd.read_csv('data/sample_submission.csv')

    sub['label'] = sub['Image_Label'].apply(lambda x: x.split('_')[1])
    sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])

    test_ids = sub['Image_Label'].apply(
        lambda x: x.split('_')[0]).drop_duplicates().values

    test_dataset = CloudDataset(df=sub, datatype='test', img_ids=test_ids, transforms=get_test_augmentation(
    ), preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=8)

    # load best saved checkpoint
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    device_ids = [0, 1]
    model = model.cuda(device_ids[0])
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(torch.load(weights_path)["model_state_dict"])

    encoded_pixels = tta_forward(test_loader, model)

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv('submission.csv', columns=[
               'Image_Label', 'EncodedPixels'], index=False)


if __name__ == '__main__':
    main()
