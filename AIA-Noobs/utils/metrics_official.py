# from official repo
# sarrarp50.metrics.segmentation import mNSD, mIoU
# sarrarp50.metrics.action_recognition import accuracy, f1k

from pickletools import uint8
import torch
import monai
import cv2
from pathlib import Path
import numpy as np
import warnings


def save_one_hot(root_dir, oh):
    # function to sotre a one hot tensor as separate images
    for i, c in enumerate(oh):
        print(c.numpy().shape)
        cv2.imwrite(f'{root_dir}/{i}.png', c.numpy().astype(np.uint8) * 255)
    exit()


def imread_one_hot(filepath, n_classes):
    # reads a segmentation mask stored as png and returns in in one-hot torch.tensor format
    img = cv2.imread(str(filepath))
    if img is None:
        raise FileNotFoundError(filepath)
    if len(img.shape) == 3:  # if the segmentation mask was 3 channel, only keep the first
        img = img[..., 0]
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).requires_grad_(False)
    return monai.networks.utils.one_hot(img, n_classes, dim=1)


def get_val_func(metric, n_classes=9, fix_nans=False):
    # this function is intented for wrapping the meanIoU and meanNSD metric computation functions
    # It returns a error computation function that is able to parse reference
    # and prediction segmentaiton samples in directory level.
    def f(dir_pred, dir_ref):
        seg_ref_paths = sorted(list(dir_ref.iterdir()))
        dir_pred = Path(dir_pred)

        acc = []
        with torch.no_grad():
            for seg_ref_p in seg_ref_paths:

                # load segmentation masks as one_hot torch tensors
                try:
                    ref = imread_one_hot(seg_ref_p, n_classes=n_classes + 1)
                except FileNotFoundError:
                    raise
                try:
                    pred = imread_one_hot(dir_pred / seg_ref_p.name, n_classes=n_classes + 1)
                except FileNotFoundError as e:
                    # if the prediciton file was not found, set all scores to zero and continue
                    acc.append([0] * n_classes)
                    continue

                if fix_nans:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        err = metric(pred, ref)
                    # if both reference and predictions are zero, set the prediciton values to one
                    # this is required for NSD because otherise values are goint to
                    # be set to nan even though the prediciton is correct.
                    # in case either the pred or corresponding ref channel is zero
                    # NSD will resurn either 0 or nan and in those cases nan is
                    # converted to zero
                    # find the zero channels in both ref and pred and create a mask
                    # in the size of the final prediction.(1xn_channels)
                    r_m, p_m = ref.mean(axis=(2, 3)), pred.mean(axis=(2, 3))
                    mask = ((r_m == 0) * (p_m == 0))[:, 1:]

                    # set the scores in cases where both ref and pred were full zero
                    #  to 1.
                    err[mask == True] = 1
                    # in cases where either was full zero but the other wasn't
                    # (score is nan ) set the corresponding score to 0
                    err.nan_to_num_()
                else:
                    err = metric(pred, ref)
                acc.append(err.detach().reshape(-1).tolist())
        return np.array(acc)  # need to add axis and then multiply with the channel scales

    return f


def mIoU(dir_pred, dir_ref, n_classes=9):
    metric = monai.metrics.MeanIoU(include_background=False, reduction='mean', get_not_nans=False, ignore_empty=False)
    validation_func = get_val_func(metric, n_classes=n_classes)
    return validation_func(dir_pred / 'segmentation', dir_ref / 'segmentation')


def mNSD(dir_pred, dir_ref, n_classes=9, channel_tau=[1] * 9):
    metric = monai.metrics.SurfaceDiceMetric(channel_tau,
                                             include_background=False,
                                             reduction='mean')
    validation_func = get_val_func(metric, n_classes=n_classes, fix_nans=True)
    return validation_func(dir_pred / 'segmentation', dir_ref / 'segmentation')

def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split

def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def _accuracy(P, Y, **kwargs):
    def acc_(p,y):
        return np.mean(p==y)
    if type(P) == list:
        return np.mean([np.mean(P[i]==Y[i]) for i in range(len(P))])
    else:
        return acc_(P,Y)


def _f1k(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p,y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        # Remove background labels
        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        # We keep track of the per-class TP, and FP.
        # In the end we just sum over them though.
        TP = np.zeros(n_classes, np.float)
        FP = np.zeros(n_classes, np.float)
        true_used = np.zeros(n_true, np.float)

        for j in range(n_pred):
            # Compute IoU against all others
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            # Get the best scoring segment
            idx = IoU.argmax()

            # If the IoU is high enough and the true segment isn't already used
            # Then it is a true positive. Otherwise is it a false positive.
            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1

        TP = TP.sum()
        FP = FP.sum()
        # False negatives are any unused true segment (i.e. "miss")
        FN = n_true - true_used.sum()

        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F1 = 2 * (precision*recall) / (precision+recall)

        # If the prec+recall=0, it is a NaN. Set these to 0.
        F1 = np.nan_to_num(F1)

        return F1

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)



def accuracy(pred_dir, ref_dir):
    ref_action = np.genfromtxt(ref_dir/'action_discrete.txt', delimiter=',')
    pred_action = np.genfromtxt(pred_dir/'action_discrete.txt', delimiter=',')
    return _accuracy(pred_action[:,1],ref_action[:,1])

    # check if they have the same length and if they do not

def f1k(pred_dir, ref_dir, k=10, n_classes=8):
    ref_action = np.genfromtxt(ref_dir/'action_discrete.txt', delimiter=',').astype(int)
    pred_action = np.genfromtxt(pred_dir/'action_discrete.txt', delimiter=',').astype(int)
    return _f1k(pred_action[:,1],ref_action[:,1], n_classes=n_classes, overlap=k/100)