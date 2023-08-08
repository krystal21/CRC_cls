# -*- coding: utf-8 -*- 
"""
时间：2022年05月20日
"""
import os
import pandas as pd
import cv2
import numpy as np
import torchvision.models as models
from torch import nn
import torch
from torchvision import transforms
import torch
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, recall_score, roc_auc_score, \
    precision_score

transform_list_val1 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

transformList2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def show_message(y_tru, y_p):
    auc = roc_auc_score(y_tru, y_p)
    y_p = (y_p >= 0.5) + 0
    acc = accuracy_score(y_tru, y_p)
    recall = recall_score(y_tru, y_p, pos_label=1)
    recall0 = recall_score(y_tru, y_p, pos_label=0)
    per = precision_score(y_tru, y_p, pos_label=1)
    per0 = precision_score(y_tru, y_p, pos_label=0)
    confu = confusion_matrix(y_tru, y_p)
    return acc, auc, recall, recall0, per, per0, confu


def get_images_labels(idx, img, lb, path_val):
    images = []
    labels = []
    for i in range(len(idx)):
        path = path_val[i]
        if img[i] == "_":
            if os.path.exists(path + str(idx[i]) + '.bmp'):
                temp = path + str(idx[i]) + '.bmp'
            elif os.path.exists(path + str(idx[i]) + '.jpg'):
                temp = path + str(idx[i]) + '.jpg'
            else:
                temp = path + str(idx[i]) + '.png'
            images.append(temp)
            labels.append(lb[i])
        else:
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


def get_model(model_name):
    if model_name == 'resnet':
        resnet101 = models.resnet101(pretrained=False)
        fc_features = resnet101.fc.in_features
        resnet101.fc = nn.Linear(fc_features, 1)
        return resnet101
    if model_name == 'resnet18':
        resnet101 = models.resnet18(pretrained=False)
        fc_features = resnet101.fc.in_features
        resnet101.fc = nn.Linear(fc_features, 1)
        return resnet101
    if model_name == 'densenet':
        densenet = models.densenet161(pretrained=False)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, 1)
        return densenet
    if model_name == 'resnext':
        resnext101_32x8d = models.resnext101_32x8d(pretrained=False)
        fc_features = resnext101_32x8d.fc.in_features
        resnext101_32x8d.fc = nn.Linear(fc_features, 1)
        return resnext101_32x8d
    if model_name == 'inception':
        inception = models.inception_v3(pretrained=False)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, 1)
        return inception
    if model_name == 'mobilenet':
        mobilenet = models.mobilenet_v2(pretrained=False)
        num_ftrs = 1280
        mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
        )
        return mobilenet
    if model_name == 'efficientnet/0':
        efficientnet = models.efficientnet_b1(pretrained=False)
        num_ftrs = 1280
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
        )
        return efficientnet


hos = ['兰州大学第二医院', '曙光医院', '运城第一医院', '余姚市人民医院', '重庆市中医院']
hos1 = ['lanzhou', 'shuguang', 'yuncheng', 'yuyao', 'chongqing']

for i, hos_name in enumerate(hos):
    save = []
    ts = []
    model_str = 'densenet'
    model = get_model(model_str)
    model_path = '/Modeldata/fold/0/0/epoch39.pth'
    path = '/home/zhangzhengjie/data/zhangzhengjie/data/' + hos1[i] + '/'
    save_path = 'result/0/'
    loaded_model = torch.load(model_path)
    model.load_state_dict(loaded_model['state_dict'], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    df = pd.read_csv('Data/ubuntu/predict/' + hos + '.csv')
    images = [path + im for im in df['path'].tolist()]
    labels = df['label'].tolist()

    y_p = None
    model.eval()
    for img in images:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.array(image))
        image = transform_list_val1(image)
        image = transformList2(image)
        image = np.expand_dims(image, 0)
        image = torch.FloatTensor(image)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            combine = model(image)
        y_pre = combine.sigmoid()
        y_pre = y_pre.reshape(-1).detach().cpu().numpy()
        if y_p is None:
            y_p = np.array(y_pre)
        else:
            y_p = np.hstack((y_p, np.array(y_pre)))

    acc, auc, recall, recall0, per, per0, confu = show_message(labels, y_p)
    ts.append([confu[0, 0], confu[0, 1], confu[1, 0], confu[1, 1]])
    temp = [hos, acc, auc, recall, recall0, per, per0, confu[0, 0], confu[0, 1], confu[1, 0], confu[1, 1]]
    save.append(temp)
    df1 = pd.DataFrame(data=save)
    df1.to_csv(save_path + hos1 + '.csv')
