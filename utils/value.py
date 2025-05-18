import cv2
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
from einops import repeat
import sys


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

def get_boxes(pred, gt):
    pred_box = excute_points(pred)
    _, _, label_box, _ = cv2.connectedComponentsWithStats(gt.astype(np.int8), connectivity=8)
    pred_boxes = convert_boxes(pred_box)[1:,:]
    label_boxes = convert_boxes(label_box)[1:,:]
    return pred_boxes, label_boxes

def calculate_metric_percase(pred, gt, i):
    idx = i
    pred = pred.astype(int) == i
    gt = gt.astype(int) == i
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    treshold_ls = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95]
    ap595 = np.average(np.array([compute_ap(pred, gt, treshold) for treshold in treshold_ls]))
    ap = compute_ap(pred, gt, 0.5)
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        pred_boxes, label_boxes = get_boxes(pred, gt)
        iou = np.mean(np.max(cal_iou(pred_boxes, label_boxes), axis=0))
    elif pred.sum() > 0 and gt.sum() == 0:
        dice = 0
        hd95 = 0
        iou = 0
    elif pred.sum() == 0 and gt.sum() > 0:
        dice = 0
        hd95 = 0
        iou = 0
    else:
        return None

    return [dice, hd95, iou, 0.5, ap, ap595, idx]


def convert_boxes(boxes):
    "左上到右下的点"
    x, y, w, h, n = boxes.T
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return np.stack((x1, y1, x2, y2), axis=1)


def cal_iou(pred_boxes, label_boxes):

    inter_x1 = np.maximum(pred_boxes[:, 0][:, None], label_boxes[:, 0])
    inter_y1 = np.maximum(pred_boxes[:, 1][:, None], label_boxes[:, 1])
    inter_x2 = np.minimum(pred_boxes[:, 2][:, None], label_boxes[:, 2])
    inter_y2 = np.minimum(pred_boxes[:, 3][:, None], label_boxes[:, 3])
    inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    true_area = (label_boxes[:, 2] - label_boxes[:, 0]) * (label_boxes[:, 3] - label_boxes[:, 1])
    union_area = pred_area[:, None] + true_area - inter_area
    
    iou = inter_area / union_area
    # print('iou', iou)
    # sys.exit()
    if iou.size == 0:
        return 0
    else:
        return iou


def evaluate_image(preds, truths, iou_threshold):

    pred_boxes, true_boxes = get_boxes(preds, truths)
    num_pred = pred_boxes.shape[0] if pred_boxes.size > 0 else 0
    num_true = true_boxes.shape[0] if true_boxes.size > 0 else 0
    
    if num_pred == 0 or num_true == 0:
        return 0, num_pred
    
    iou_matrix = cal_iou(pred_boxes, true_boxes)
    matched_true = np.zeros(num_true, dtype=bool)
    tp = 0
    
    for pred_idx in range(num_pred):
        best_iou = np.max(iou_matrix[pred_idx])
        best_true_idx = np.argmax(iou_matrix[pred_idx])
        
        if best_iou >= iou_threshold and not matched_true[best_true_idx]:
            tp += 1
            matched_true[best_true_idx] = True
    
    return tp, num_pred


def compute_ap(preds, truths, iou_threshold):
    total_tp = 0
    total_pred = 0
        
    tp, num_pred = evaluate_image(preds, truths, iou_threshold)
    total_tp += tp
    total_pred += num_pred
    
    return total_tp / total_pred if total_pred > 0 else 0.0


def excute_points(pred, threshold_area=100):
    _, _, pred_box, _ = cv2.connectedComponentsWithStats(pred.astype(np.int8), connectivity=8)
    box_label = []

    for i in range(len(pred_box)):
        if pred_box[i][2]*pred_box[i][3] > threshold_area:
            box_label.append(pred_box[i])
    
    return np.array(box_label)


def dilated_mask(predtion):
    # k = np.array(
    #     [[0, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 0],
    #      [1, 1, 1, 1, 1],
    #      [0, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 0]], dtype=np.uint8)
    k = np.array(
        [[1, 0, 0, 0, 1],
         [0, 1, 0, 1, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 0, 0, 0, 1]], dtype=np.uint8)
    dilated_predition = cv2.dilate(predtion.astype(np.float32), k, iterations=1)
    return dilated_predition

def erode_mask(predtion):
    # k = np.array(
    #     [[1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1],
    #      [1, 1, 1, 1, 1]], dtype=np.uint8)
    k = np.array(
        [[0, 1, 0],
         [1, 1, 0],
         [0, 1, 0],], dtype=np.uint8)
    # k = np.array(
    #     [[0, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 0],
    #      [1, 1, 1, 1, 1],
    #      [0, 0, 1, 0, 0],
    #      [0, 0, 1, 0, 0]], dtype=np.uint8)
    # k = np.array(
    #     [[1, 0, 0, 0, 1],
    #      [0, 1, 0, 1, 0],
    #      [0, 0, 1, 0, 0],
    #      [0, 1, 0, 1, 0],
    #      [1, 0, 0, 0, 1]], dtype=np.uint8)
    erode_predition = cv2.erode(predtion.astype(np.float32), k, iterations=1)
    return erode_predition

def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        x, y = image.shape[0], image.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y, 1), order=3)  # previous using 0, patch_size[0], patch_size[1]
        inputs = np.transpose(image, (2, 0, 1))
        inputs = torch.from_numpy(inputs).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            # out = torch.softmax(output_masks, dim=1).cpu().detach().numpy()
            # out = out[0,1,:,:]
            # out = (out > 0.3).astype(int)
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction = pred
    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        prediction = erode_mask(prediction)
        result = calculate_metric_percase(prediction, label, i)
        if result is not None:
            metric_list.append(result)

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction)
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list
