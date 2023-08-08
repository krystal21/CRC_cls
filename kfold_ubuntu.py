# -*- coding: utf-8 -*- 
"""
时间：2022年07月09日
"""
import os
from parameter import parse_opts

args = parse_opts()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
from sklearn.model_selection import train_test_split
from utils.dataloader1 import DatasetGenerator
import torchvision.models as models
from torch import nn
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from utils.trainer import train_step, validation_step
import numpy as np
import cv2
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import pandas as pd


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
    if model_name == 'resnet152':
        resnet101 = models.resnet152(pretrained=True)
        fc_features = resnet101.fc.in_features
        resnet101.fc = nn.Linear(fc_features, 1)
        return resnet101
    if model_name == 'resnet101':
        resnet101 = models.resnet101(pretrained=True)
        fc_features = resnet101.fc.in_features
        resnet101.fc = nn.Linear(fc_features, 1)
        return resnet101
    if model_name == 'resnet50':
        resnet50 = models.resnet50(pretrained=True)
        fc_features = resnet50.fc.in_features
        resnet50.fc = nn.Linear(fc_features, 1)
        return resnet50
    if model_name == 'densenet161':
        densenet = models.densenet161(pretrained=True)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, 1)
        return densenet
    if model_name == 'densenet121':
        densenet = models.densenet121(pretrained=True)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, 1)
        return densenet
    if model_name == 'densenet201':
        densenet = models.densenet201(pretrained=True)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, 1)
        return densenet
    if model_name == 'resnext':
        resnext101_32x8d = models.resnext101_32x8d(pretrained=True)
        fc_features = resnext101_32x8d.fc.in_features
        resnext101_32x8d.fc = nn.Linear(fc_features, 1)
        return resnext101_32x8d
    if model_name == 'inception':
        inception = models.inception_v3(pretrained=True)
        fc_features = inception.fc.in_features
        inception.fc = nn.Linear(fc_features, 1)
        return inception
    if model_name == 'mobilenet':
        mobilenet = models.mobilenet_v2(pretrained=True)
        num_ftrs = 1280
        mobilenet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
        )
        return mobilenet
    if model_name == 'efficientnet/0':
        efficientnet = models.efficientnet_b0(pretrained=True)
        num_ftrs = 1280
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
        )
        return efficientnet
    if model_name == 'efficientnet/66':
        efficientnet = models.efficientnet_b1(pretrained=True)
        num_ftrs = 1280
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
        )
        return efficientnet
    if model_name == 'efficientnet/2':
        efficientnet = models.efficientnet_b2(pretrained=True)
        num_ftrs = 1408
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
        )
        return efficientnet
    if model_name == 'efficientnet/3':
        efficientnet = models.efficientnet_b2(pretrained=True)
        num_ftrs = 1536
        efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_ftrs, 1),
        )
        return efficientnet


transform_list1 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(-180, +180)),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
])
transformList2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
transform_list_val1 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224)
])

if __name__ == "__main__":

    model_str = args.model_str

    # 参数设置
    lr = args.lr
    Batch_size = args.batch_size
    epochs = args.epochs
    save_path = args.model_path

    df1 = pd.read_csv('Data/ubuntu/良恶性/仁济良恶性.csv')
    # for hos_name in ['兰州大学第二医院', '曙光医院', '运城第一医院', '余姚市人民医院', '重庆市中医院']:
    #     df = pd.read_csv('Data/ubuntu/良恶性/' + hos_name + '良恶性.csv')
    #     df1 = pd.concat([df1, df], ignore_index=True)
    idx_patient = df1['编号'].tolist()
    idx_image = df1['图片编号'].tolist()
    labels = df1['良恶性'].tolist()
    path = df1['path'].tolist()

    skf = StratifiedKFold(n_splits=5, random_state=6, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(idx_patient, labels)):
        model = get_model(model_str)
        model.cuda()
        if ',' in args.gpu_id:
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5,
                                                               verbose=True)
        criterion = torch.nn.BCEWithLogitsLoss()

        idx_patient_val = np.array(idx_patient)[val_idx]
        labels_v = np.array(labels)[val_idx]
        idx_image_val = np.array(idx_image)[val_idx]
        path_v = np.array(path)[val_idx]

        idx_patient_train = np.array(idx_patient)[train_idx]
        labels_t = np.array(labels)[train_idx]
        idx_image_train = np.array(idx_image)[train_idx]
        path_t = np.array(path)[train_idx]

        images_train, labels_train = get_images_labels(idx_patient_train, idx_image_train, labels_t, path_t)
        images_val, labels_val = get_images_labels(idx_patient_val, idx_image_val, labels_v, path_v)

        data_train = DatasetGenerator(image_names=images_train, labels=labels_train,
                                      transform1=transform_list1,
                                      transform2=transformList2, set='train')
        train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=Batch_size,
                                                   shuffle=True, num_workers=args.num_workers, pin_memory=True)

        data_val = DatasetGenerator(image_names=images_val, labels=labels_val,
                                    transform1=transform_list_val1,
                                    transform2=transformList2, set='val')
        val_loader = torch.utils.data.DataLoader(dataset=data_val, batch_size=1,
                                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)

        list_train = []
        list_val = []
        best = 0.8
        for epoch in range(epochs):
            epoch_loss_train, acc_train, auc_train, recall_train, recall0_train, per_train, per0_train = \
                train_step(train_loader, model, optimizer, criterion, epoch, epochs)
            epoch_loss_val, acc_val, auc_val, recall_val, recall0_val, per_val, per0_val = \
                validation_step(val_loader, model, criterion, epoch, epochs)
            result_train = [acc_train, auc_train, recall_train, recall0_train, per_train, per0_train]
            result_val = [acc_val, auc_val, recall_val, recall0_val, per_val, per0_val]
            list_train.append(result_train)
            list_val.append(result_val)
            if acc_val > best:
                best = acc_val
                model_save_file = os.path.join(save_path + str(fold) + '/epoch' + str(epoch) + '.pth')
                torch.save(model, model_save_file)
            scheduler.step(epoch_loss_train)
            data_fm = pd.DataFrame({'acc_train': list(np.array(list_train).T[0]),
                                    'auc_train': list(np.array(list_train).T[1]),
                                    'recall_train': list(np.array(list_train).T[2]),
                                    'recall0_train': list(np.array(list_train).T[3]),
                                    'per_train': list(np.array(list_train).T[4]),
                                    'per0_train': list(np.array(list_train).T[5]),
                                    'acc_val': list(np.array(list_val).T[0]),
                                    'auc_val': list(np.array(list_val).T[1]),
                                    'recall_val': list(np.array(list_val).T[2]),
                                    'recall0_val': list(np.array(list_val).T[3]),
                                    'per_val': list(np.array(list_val).T[4]),
                                    'per0_val': list(np.array(list_val).T[5]),
                                    })
            data_fm.to_csv(args.results+str(fold) + '/result.csv')
