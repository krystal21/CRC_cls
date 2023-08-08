# -*- coding: utf-8 -*- 
"""
时间：2022年05月20日
"""
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_score
from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt
import matplotlib
import torchvision.models as models
from torchsummary import summary
from torch import nn
import torch
from sklearn.metrics import roc_curve, auc

np.random.seed(888)


def show_message(y_tru, y_p):
    auc = roc_auc_score(y_tru, y_p)
    y_p = (y_p >= 0.5) + 0
    acc = accuracy_score(y_tru, y_p)
    recall = recall_score(y_tru, y_p, pos_label=1)
    recall0 = recall_score(y_tru, y_p, pos_label=0)
    per = precision_score(y_tru, y_p, pos_label=1)
    per0 = precision_score(y_tru, y_p, pos_label=0)
    return acc, auc, recall, recall0, per, per0


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def bootstrap_auc(pred, y, nsamples=1000):
    auc_values = []
    acc_values = []
    r0_values = []
    r1_values = []
    p0_values = []
    p1_values = []

    pre1 = np.array(pred)
    label = np.array(y)
    n = len(label)
    z = n * 5
    auc = roc_auc_score(label, pre1)
    y_p = (pre1 >= 0.5) + 0
    y_tru = label
    acc2 = accuracy_score(y_tru, y_p)
    recall2 = recall_score(y_tru, y_p, pos_label=1)
    recall02 = recall_score(y_tru, y_p, pos_label=0)
    per2 = precision_score(y_tru, y_p, pos_label=1)
    per02 = precision_score(y_tru, y_p, pos_label=0)
    for b in range(nsamples):
        index_arr = np.random.randint(0, n, size=z)
        # index_arr = np.random.randint(0, n, size=n)
        pre_s = pre1[index_arr]
        label1 = label[index_arr]

        roc_auc = roc_auc_score(label1, pre_s)
        auc_values.append(roc_auc)

        y_p = (pre_s >= 0.5) + 0
        y_tru = label1
        acc = accuracy_score(y_tru, y_p)
        acc_values.append(acc)
        # recall = recall_score(y_tru, y_p, pos_label=1)
        # r1_values.append(recall)
        # recall0 = recall_score(y_tru, y_p, pos_label=0)
        # r0_values.append(recall0)
        # per = precision_score(y_tru, y_p, pos_label=1)
        # p1_values.append(per)
        # per0 = precision_score(y_tru, y_p, pos_label=0)
        # p0_values.append(per0)

    values = acc_values
    v = acc2
    # values = auc_values
    # v = auc
    values.sort()
    l, h = np.percentile(values, (2.5, 97.5))
    print(hos, v, l, h, str(round(v, 4)) + '(' + str(round(l, 4)) + '-' + str(round(h, 4)) + ')')
    return np.percentile(values, (2.5, 97.5))


if __name__ == "__main__":
    image_path = "D:\\DATA\\rj20220510\\cropped\\"
    x = "D:\\DATA\\rj\\all\\"
    u = '/home/zhangzhengjie/data/zhangzhengjie/data/rj/'
    csv_path = 'Data/NewData/NBI/良恶性/'
    save_path = 'Data/ubuntu/良恶性/'
    hos_names = ['仁济', 'all', '兰州大学第二医院', '曙光医院', '运城第一医院', '重庆市中医院', '余姚市人民医院']
    HN = ['Internal Validation(RJH)', 'External Validation(tatal)', 'LZSH', 'SGH', 'YCFH', 'YYH', 'CQCMH']
    HN = ['Internal Validation(RJH)', 'External Validation(tatal)', 'LZSH', 'SGH', 'YCFH', 'YYH', 'CQCMH']
    # hos_names = ['兰州大学第二医院', '曙光医院', '运城第一医院', '重庆市中医院', '余姚市人民医院', '仁济']
    hos_path = ['lanzhou', 'shuguang', 'yuncheng', 'chongqing', 'chengdu', 'yuyao']
    path = "D:\\DATA\\rj20220510\\z\\"
    # csv_path = '82k/152/densenet161/'
    i = 0
    flag = 20
    if flag == 20:
        for I, hos in enumerate(hos_names):
            df = pd.read_csv('z/' + hos + '.csv')
            y_tru = np.array(df['lable'].tolist())
            y_p = np.array(df['pre'].tolist())
            fpr, tpr, thersholds = roc_curve(y_tru, y_p, pos_label=1)
            print(len(thersholds))
            roc_auc = auc(fpr, tpr)
            if hos == '仁济' or hos == 'all':
                plt.plot(fpr, tpr, label=HN[I] + '\n(AUC = {0:.4f})'.format(roc_auc), lw=1.5)
            else:
                plt.plot(fpr, tpr, label=HN[I] + '(AUC = {0:.4f})'.format(roc_auc), lw=1.5)

        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Specificity')
        plt.ylabel('Sensitivity')  # 可以使用中文，但需要导入一些库即字体
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.gca().set_aspect(1)
        # plt.savefig("test.svg", dpi=600, format="svg")
        plt.show()

    if flag == 19:
        for hos in hos_names:
            df = pd.read_csv('z/' + hos + '良恶性m.csv')
            df1 = pd.read_csv('z/' + hos + '.csv')
            print(sum(df['良恶性']), len(df) - sum(df['良恶性']), len(df), sum(df1['lable']),
                  len(df1) - sum(df1['lable']), len(df1))

    if flag == 18:
        x = []
        y = []
        for hos in hos_names:
            df = pd.read_csv('z/' + hos + '.csv')
            y_tru = np.array(df['lable'].tolist())
            y_p = np.array(df['pre'].tolist())
            acc, auc, recall, recall0, per, per0, confu = show_message(y_tru, y_p, 0.5)
            print(hos, acc, auc, recall, recall0, per, per0)
            if hos != '仁济':
                x.extend(df['lable'].tolist())
                y.extend(df['pre'].tolist())
        df1 = pd.DataFrame({'pre': y, 'lable': x})
        df1.to_csv('z/all.csv')
        acc, auc, recall, recall0, per, per0, confu = show_message(np.array(x), np.array(y), 0.5)
        print(acc, auc, recall, recall0, per, per0)

    if flag == 17:
        pp = []
        ll = []
        for hos in hos_names:
            df = pd.read_csv('z/' + hos + '.csv')
            pre = df['pre'].tolist()
            label = df['lable'].tolist()
            if hos != '仁济':
                pp.extend(pre)
                ll.extend(label)
            # auc = roc_auc_score(label, pre)
            #
            l, h = bootstrap_auc(pre, label, 1000)
            # print(hos,auc,l,h,str(round(auc,4))+'('+str(round(l,4))+'-'+str(round(h,4))+')')

        auc = roc_auc_score(ll, pp)
        l, h = bootstrap_auc(pp, ll, 1000)
        # print('测试', auc, l, h, str(round(auc,4))+'(' + str(round(l, 4)) + '-' + str(round(h, 4)) + ')')
