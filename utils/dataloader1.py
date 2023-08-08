# -*- coding: utf-8 -*- 
"""
时间：2022年03月17日
"""
# encoding: utf-8
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


def get_images_labels(idx, img, lb, path):
    images = []
    labels = []
    for i in range(len(idx)):
        for j in img[i]:
            if os.path.exists(path + str(idx[i]) + j + '.bmp'):
                temp = path + str(idx[i]) + j + '.bmp'
            elif os.path.exists(path + str(idx[i]) + j + '.jpg'):
                temp = path + str(idx[i]) + j + '.jpg'
            else:
                temp = path + str(idx[i]) + j + '.png'
            images.append(temp)
            labels.append(lb[i])
    return images, labels


class DatasetGenerator(Dataset):
    def __init__(self, image_names, labels, transform1=None, transform2=None, set='train'):
        self.image_names = image_names
        self.labels = labels
        self.transform1 = transform1
        self.transform2 = transform2
        self.image_names = image_names
        self.labels = labels
        self.set = set

    def __getitem__(self, index):
        image_name = self.image_names[index]
        label = self.labels_ag[index]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.array(image))
        if self.transform1 is not None:
            image = self.transform1(image)
        if self.transform2 is not None:
            image = self.transform2(image)
        return torch.FloatTensor(image), label

    def __len__(self):
        return len(self.image_names_ag)
