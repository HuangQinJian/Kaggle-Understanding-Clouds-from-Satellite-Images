'''
@Description:
@Author: HuangQinJian
@Date: 2019-06-19 19:11:44
@LastEditTime: 2019-10-06 11:39:30
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

import albumentations as albu
import segmentation_models_pytorch as smp
from catalyst.dl import utils
from catalyst.dl.callbacks import (CheckpointCallback, DiceCallback,
                                   EarlyStoppingCallback, InferCallback,
                                   IouCallback)
from catalyst.dl.runner import SupervisedRunner
from clouddataset import CloudDataset
from nadam import Nadam
from sklearn.model_selection import train_test_split
from tqdm import tqdm

plt.switch_backend('agg')

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
MODEL_NAME = 'FPN'

CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']
ACTIVATION = 'softmax'

weights_per_class = torch.FloatTensor([1, 1, 1, 1]).cuda()

WIDTH_TRAIN = 1280
HEIGHT_TRAIN = 640


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0,
                              shift_limit=0.1, p=0.5, border_mode=0),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
        albu.Resize(HEIGHT_TRAIN, WIDTH_TRAIN)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
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


def train_model(epoch, train_loader, valid_loader, valid_dataset, log_dir):
    # create segmentation model with pretrained encoder

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )

    loss = smp.utils.losses.BCEDiceLoss()

    optimizer = Nadam(model.parameters(), lr=1e-5)
    model = nn.DataParallel(model)
    # optimizer = torch.optim.Adam([{'params': model.module.decoder.parameters(), 'lr': 1e-4},
    #                               # decrease lr for encoder in order not to permute
    #                               # pre-trained weights with large gradients on training start
    #                               {'params': model.module.encoder.parameters(), 'lr': 1e-6}, ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epoch // 9) + 1)

    runner = SupervisedRunner()

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    runner.train(
        model=model,
        criterion=loss,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), IouCallback(), EarlyStoppingCallback(
            patience=6, min_delta=0.001)],
        logdir=log_dir,
        num_epochs=epoch,
        verbose=True
    )

    probabilities, valid_masks = valid_model(
        runner, model, valid_loader, valid_dataset,  log_dir)

    get_optimal_thres(probabilities, valid_masks)


def valid_model(runner, model, valid_loader, valid_dataset, log_dir):
    encoded_pixels = []
    loaders = {"infer": valid_loader}
    runner.infer(
        model=model,
        loaders=loaders,
        callbacks=[CheckpointCallback(resume=log_dir+'/checkpoints/best.pth'),
                   InferCallback()
                   ],
    )
    valid_masks = []
    probabilities = np.zeros((2220, HEIGHT_TRAIN, WIDTH_TRAIN))
    for i, (batch, output) in enumerate(tqdm(zip(valid_dataset, runner.callbacks[0].predictions["logits"]))):
        image, mask = batch
        for m in mask:
            if m.shape != (HEIGHT_TRAIN, WIDTH_TRAIN):
                m = cv2.resize(m, dsize=(WIDTH_TRAIN, HEIGHT_TRAIN),
                               interpolation=cv2.INTER_LINEAR)
            valid_masks.append(m)

        for j, probability in enumerate(output):
            if probability.shape != (HEIGHT_TRAIN, WIDTH_TRAIN):
                probability = cv2.resize(probability, dsize=(
                    WIDTH_TRAIN, HEIGHT_TRAIN), interpolation=cv2.INTER_LINEAR)
            probabilities[i * 4 + j, :, :] = probability
    return probabilities, valid_masks


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((HEIGHT_TRAIN, WIDTH_TRAIN), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def dice(img1, img2):
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


def get_optimal_thres(probabilities, valid_masks):
    class_params = {}
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    for class_id in range(4):
        print(class_id)
        attempts = []
        for t in range(50, 100, 5):
            t /= 100
            for ms in [6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000]:
                masks = []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(
                        sigmoid(probability), t, ms)
                    masks.append(predict)
                d = []
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))
                # print(t, ms, np.mean(d))
                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(
            attempts, columns=['threshold', 'size', 'dice'])

        attempts_df = attempts_df.sort_values('dice', ascending=False)
        print(attempts_df.head())
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        class_params[class_id] = (best_threshold, best_size)
    print(class_params)


def get_arguments():
    parser = argparse.ArgumentParser(description="data_prepare")
    parser.add_argument("--train_csv_path", type=str, required=True,
                        help="the path of the train_csv")
    parser.add_argument("--epoch", type=int, required=True,
                        help="epoch")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="log_dir")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="batch_size")
    return parser.parse_args()


def main():
    args = get_arguments()
    train_csv_path = args.train_csv_path
    epoch = args.epoch
    log_dir = args.log_dir
    batch_size = args.batch_size

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    train = pd.read_csv(train_csv_path)
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
        reset_index().rename(
            columns={'index': 'img_id', 'Image_Label': 'count'})
    train_ids, valid_ids = train_test_split(
        id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)

    print(len(train_ids), len(valid_ids))

    train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms=get_training_augmentation(
    ), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms=get_validation_augmentation(
    ), preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    train_model(epoch, train_loader, valid_loader, valid_dataset, log_dir)


if __name__ == '__main__':
    main()
