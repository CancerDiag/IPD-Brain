from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import numpy as np

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def roc_threshold(label, prediction):
    fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    c_auc = roc_auc_score(label, prediction)
    return c_auc, threshold_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def eval_metric_(oprob, label):
    ## convert to numpy
    oprob = oprob.cpu().detach().numpy()
    label = label.cpu().detach().numpy()


    # label_onehot = LabelBinarizer().fit_transform(label)
    auc = roc_auc_score(label, oprob, multi_class="ovo", average="macro")
    
    y_pred = np.argmax(oprob, axis = 1)
    accuracy = accuracy_score(label, y_pred)

    precision = precision_score(label, y_pred, average='macro')
    recall = recall_score(label, y_pred, average='macro')
    F1 = f1_score(label, y_pred, average='macro')
    return accuracy, precision, recall, F1, auc


def eval_metric(oprob, label):
    oprob = oprob[:, -1]

    auc, threshold = roc_threshold(label.cpu().numpy(), oprob.detach().cpu().numpy())
    prob = oprob > threshold
    label = label > threshold

    TP = (prob & label).sum(0).float()
    TN = ((~prob) & (~label)).sum(0).float()
    FP = (prob & (~label)).sum(0).float()
    FN = ((~prob) & label).sum(0).float()

    accuracy = torch.mean(( TP + TN ) / ( TP + TN + FP + FN + 1e-12))
    precision = torch.mean(TP / (TP + FP + 1e-12))
    recall = torch.mean(TP / (TP + FN + 1e-12))
    # specificity = torch.mean( TN / (TN + FP + 1e-12))
    F1 = 2*(precision * recall) / (precision + recall+1e-12)

    return accuracy, precision, recall, F1, auc