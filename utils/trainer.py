# -*- coding: utf-8 -*- 
"""
时间：2022年03月17日
"""
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, recall_score, roc_auc_score, precision_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_message(y_tru, y_p):
    auc = roc_auc_score(y_tru, y_p)
    y_p = (y_p >= 0.5) + 0
    acc = accuracy_score(y_tru, y_p)
    recall = recall_score(y_tru, y_p, pos_label=1)
    recall0 = recall_score(y_tru, y_p, pos_label=0)
    per = precision_score(y_tru, y_p, pos_label=1)
    per0 = precision_score(y_tru, y_p, pos_label=0)
    return acc, auc, recall, recall0, per, per0


def train_step(train_loader, model, optimizer, criterion, epoch, epochs):
    model.train()
    epoch_loss = 0.0
    iters_per_epoch = len(train_loader)
    y_tru = None
    y_p = None
    with tqdm(total=iters_per_epoch, desc=f'Train_Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        for step, (images, labels) in enumerate(train_loader):
            if y_tru is None:
                y_tru = np.array(labels)
            else:
                y_tru = np.hstack((y_tru, np.array(labels)))
            # print(labels)
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.float().cuda()
                labels = labels.unsqueeze(1)
            combine = model(images)
            lossvalue = criterion(combine, labels)
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            y_pre = combine.sigmoid()
            y_pre = y_pre.reshape(-1).detach().cpu().numpy()
            if y_p is None:
                y_p = np.array(y_pre)
            else:
                y_p = np.hstack((y_p, np.array(y_pre)))

            epoch_loss += lossvalue.item()

            pbar.set_postfix(**{'epoch_loss': epoch_loss / (step + 1),
                                'lr': optimizer.param_groups[0]['lr'],
                                })
            pbar.update(1)

        acc, auc, recall, recall0, per, per0 = show_message(y_tru, y_p, 0.5)
        epoch_loss = epoch_loss / iters_per_epoch
        pbar.set_postfix(**{'acc': acc,
                            'auc': auc,
                            'recall': recall,
                            'recall0': recall0,
                            })
        pbar.update(1)
        return epoch_loss, acc, auc, recall, recall0, per, per0


def validation_step(val_loader, model, criterion, epoch, epochs):
    # switch to train mode
    model.eval()
    epoch_loss = 0.0

    iters_per_epoch = len(val_loader)

    y_tru = None
    y_p = None
    with tqdm(total=iters_per_epoch, desc=f'Val_Epoch {epoch + 1}/{epochs}', postfix=dict, mininterval=0.3) as pbar:
        for step, (images, labels) in enumerate(val_loader):
            if y_tru is None:
                y_tru = np.array(labels)
            else:
                y_tru = np.hstack((y_tru, np.array(labels)))

            if torch.cuda.is_available():
                images = images.cuda()

            if torch.cuda.is_available():
                labels = labels.float().cuda()
                # labels = labels.reshape(labels.shape[0], 1)
                labels = labels.unsqueeze(1).to(device)
            with torch.no_grad():
                combine = model(images)
            lossValue = criterion(combine, labels)
            y_pre = combine.sigmoid()
            y_pre = y_pre.reshape(-1).detach().cpu().numpy()
            if y_p is None:
                y_p = np.array(y_pre)
            else:
                y_p = np.hstack((y_p, np.array(y_pre)))

            epoch_loss += lossValue.item()

            pbar.set_postfix(**{'epoch_loss': epoch_loss / (step + 1),
                                })
            pbar.update(1)

        acc, auc, recall, recall0, per, per0 = show_message(y_tru, y_p)
        epoch_loss = epoch_loss / iters_per_epoch

        pbar.set_postfix(**{'acc': acc,
                            'auc': auc,
                            'recall': recall,
                            'recall0': recall0,
                            })
        pbar.update(1)
        return epoch_loss, acc, auc, recall, recall0, per, per0


def save_file(model, epoch_loss, acc, auc, recall, recall0, model_save_file, epoch):
    torch.save({'state_dict': model.state_dict(), 'loss': epoch_loss,
                'acc': acc, 'auc': auc, 'recall': recall, 'recall0': recall0, 'epoch': epoch + 1}, model_save_file)