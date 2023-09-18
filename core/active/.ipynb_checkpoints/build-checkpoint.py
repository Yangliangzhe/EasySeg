import math
import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from .floating_region import FloatingRegionScore
from .spatial_purity import SpatialPurity
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
import cv2

def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    if len(gt_mask.shape)==4:
        intersection = np.sum(np.logical_and(np.logical_and(pred_mask.cpu().numpy(), obj_gt_mask.cpu().numpy()), ignore_gt_mask_inv.cpu().numpy()).reshape(pred_mask.shape[0],pred_mask.shape[1],-1),2)
        union = np.sum(np.logical_and(np.logical_or(pred_mask.cpu().numpy(), obj_gt_mask.cpu().numpy()), ignore_gt_mask_inv.cpu().numpy()).reshape(pred_mask.shape[0],pred_mask.shape[1],-1),2)
    elif len(gt_mask.shape)==3:
        intersection = np.sum(np.logical_and(np.logical_and(pred_mask.cpu().numpy(), obj_gt_mask.cpu().numpy()), ignore_gt_mask_inv.cpu().numpy()).reshape(pred_mask.shape[0],-1),1)
        union = np.sum(np.logical_and(np.logical_or(pred_mask.cpu().numpy(), obj_gt_mask.cpu().numpy()), ignore_gt_mask_inv.cpu().numpy()).reshape(pred_mask.shape[0],-1),1)
    else:
        intersection = np.logical_and(np.logical_and(pred_mask.cpu().numpy(), obj_gt_mask.cpu().numpy()), ignore_gt_mask_inv.cpu().numpy()).sum()
        union = np.logical_and(np.logical_or(pred_mask.cpu().numpy(), obj_gt_mask.cpu().numpy()), ignore_gt_mask_inv.cpu().numpy()).sum()

    return intersection / union

def PixelSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER))# (512*512)*(5142*512) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K
    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']

            label = tgt_data['label'] # 用于交互式选点

            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            #if origin_label.shape != 4:
                #ignore_mask = F.one_hot(torch.clamp(origin_label, 0, cfg.MODEL.NUM_CLASSES),\
                #    num_classes=cfg.MODEL.NUM_CLASSES+1).permute(0,3,1,2)[:,6,:,:]#.cuda(non_blocking=True)
                #ignore_mask = origin_label_onehot[:,6,:,:]
                #origin_label_onehot = origin_label_onehot[:,:6,:,:]
            
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):
                active_pixels_i = active_pixels
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                p = torch.softmax(output, dim=0)
                entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                pseudo_label = torch.argmax(p, dim=0)
                one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                score = entropy * purity

                ignore_mask = origin_label[i] == 255
                score[active] = -float('inf')
                score[ignore_mask] = -float('inf')

                for pixel in range(active_pixels_i):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()

class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        pc = np.diag(hist) / np.sum(hist, axis=1)
        rc = np.diag(hist) / np.sum(hist, axis=0)
        f1 = 2*np.multiply(pc, rc) / (pc + rc)
        print(f1)

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
                "F1 : \t": f1,
            },
            cls_iu,
        )

    def get_labelscores(self, label_trues, label_preds):
        # H W
        hist = self._fast_hist(label_trues.flatten(), label_preds.flatten(), self.n_classes)
        gt = np.sum(hist, 1)
        not_exist = gt==0
        gt[not_exist] = 1e6
        scores = np.max([hist/gt, hist/(gt.reshape(self.n_classes, 1))], 0) # 分母不加1，当没有某类的时候不需要点击，加一反而要点击
            
        #print(hist, hist/(gt.reshape(self.n_classes, 1)),hist/gt,scores)
        #print(np.sum(label_trues==label_preds)/(320*320),hist, scores)
        return np.max(scores - np.diag(np.diag(scores)), 1)

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

def OralPixelSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER))# (512*512)*(5142*512) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K

    labelscore = runningScore(cfg.MODEL.NUM_CLASSES)
    with torch.no_grad():
        bad_pred_class_list = np.zeros((cfg.MODEL.NUM_CLASSES,))
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            label = tgt_data['label'] # 用于交互式选点

            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            if origin_label.shape != 4:
                #origin_label_onehot = F.one_hot(origin_label, num_classes=cfg.MODEL.NUM_CLASSES).permute(0,3,1,2).cuda(non_blocking=True)
                origin_label_onehot = F.one_hot(torch.clamp(origin_label, 0, cfg.MODEL.NUM_CLASSES),\
                    num_classes=cfg.MODEL.NUM_CLASSES+1).permute(0,3,1,2)#.cuda(non_blocking=True)
                ignore_mask = origin_label_onehot[:,6,:,:]
                origin_label_onehot = origin_label_onehot[:,:6,:,:]
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):
                active_pixels_i = active_pixels
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                p = torch.softmax(output, dim=0)
                entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                pseudo_label = torch.argmax(p, dim=0)
                one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                score = entropy * purity

                score[active] = -float('inf')
                score[ignore_mask[i]] = -float('inf')
                #iou = get_iou(one_hot[0], origin_label_onehot[i])
                fn_mask = np.logical_and(origin_label_onehot[i].cpu().numpy(), np.logical_not(one_hot[i].cpu().numpy()))
                #fp_mask = np.logical_and(np.logical_not(origin_label_onehot[i].cpu().numpy()), (one_hot[i].cpu().numpy()))
                #print(fp_mask.shape)
                #for class_ in fp_mask.shape[0]: # 对每个预测类

                #    for 
                lscore = labelscore.get_labelscores(ground_truth.cpu().numpy(), pseudo_label.cpu().numpy())
                #fn_score = np.sum(fn_mask.reshape(fn_mask.shape[0],-1),1) \
                #    / (np.sum(np.logical_or(one_hot[i].cpu().numpy(), origin_label_onehot[i].cpu().numpy()).reshape(fn_mask.shape[0],-1),1)+1)
                #fn_score = np.sum(fn_mask.reshape(fn_mask.shape[0],-1),1) \
                #    / (np.sum(origin_label_onehot[i].cpu().numpy().reshape(fn_mask.shape[0],-1),1)+1)
                #print(lscore.shape)
                bad_pred_class = lscore >= cfg.INTERACTIVE.FN_THRESHOLD
                bad_pred_class_list += bad_pred_class
                #print(bad_pred_class)
                bad_pred_class_num = (bad_pred_class).sum()
                bad_pred_class_index = np.where(lscore >= cfg.INTERACTIVE.FN_THRESHOLD)[0]
                #print(fn_score, bad_pred_class_num, bad_pred_class_index)
                
                for j in range(bad_pred_class_num):
                    if active_pixels<1:
                        break
                    #bad_gt = origin_label_onehot[i][bad_pred_class_index[j]]
                    #bad_pred = one_hot[i][bad_pred_class_index[j]]
                    
                    fn_mask_dt = cv2.distanceTransform(fn_mask[bad_pred_class_index[j]].astype(np.uint8), cv2.DIST_L2, 0)
                    h, w = np.where(fn_mask_dt == np.max(fn_mask_dt))
                    
                    h = h[0]
                    w = w[0]
                    if active_mask[h, w] == 255:
                        # mask out
                        start_w = w - mask_radius if w - mask_radius >= 0 else 0
                        start_h = h - mask_radius if h - mask_radius >= 0 else 0
                        end_w = w + mask_radius + 1
                        end_h = h + mask_radius + 1
                        score[start_h:end_h, start_w:end_w] = -float('inf')
                        active[start_h:end_h, start_w:end_w] = True
                        selected[h, w] = True
                        # active sampling
                        active_mask[h, w] = ground_truth[h, w]
                        #输入才要cv2.circle(mask, (coords_y, coords_x), radius, 1, -1)
                        active_pixels_i -= 1

                for pixel in range(active_pixels_i):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])
    print(bad_pred_class_list)
    feature_extractor.train()
    classifier.train()

def MyPixelSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER))# (512*512)*(5142*512) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        bad_pred_class_list = np.zeros((cfg.MODEL.NUM_CLASSES,))
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            label = tgt_data['label'] # 用于交互式选点

            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            if origin_label.shape != 4:
                #origin_label_onehot = F.one_hot(origin_label, num_classes=cfg.MODEL.NUM_CLASSES).permute(0,3,1,2).cuda(non_blocking=True)
                origin_label_onehot = F.one_hot(torch.clamp(origin_label, 0, cfg.MODEL.NUM_CLASSES),\
                    num_classes=cfg.MODEL.NUM_CLASSES+1).permute(0,3,1,2)#.cuda(non_blocking=True)
                ignore_mask = origin_label_onehot[:,6,:,:]
                origin_label_onehot = origin_label_onehot[:,:6,:,:]
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):
                active_pixels_i = active_pixels
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                p = torch.softmax(output, dim=0)
                entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                pseudo_label = torch.argmax(p, dim=0)
                one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                score = entropy * purity

                score[active] = -float('inf')
                score[ignore_mask[i]] = -float('inf')
                #iou = get_iou(one_hot[0], origin_label_onehot[i])
                fn_mask = np.logical_and(origin_label_onehot[i].cpu().numpy(), np.logical_not(one_hot[i].cpu().numpy()))
                #fp_mask = np.logical_and(np.logical_not(origin_label_onehot[i].cpu().numpy()), (one_hot[i].cpu().numpy()))
                #print(fp_mask.shape)
                #for class_ in fp_mask.shape[0]: # 对每个预测类
                #    for 

                fn_score = np.sum(fn_mask.reshape(fn_mask.shape[0],-1),1) \
                    / (np.sum(np.logical_or(one_hot[i].cpu().numpy(), origin_label_onehot[i].cpu().numpy()).reshape(fn_mask.shape[0],-1),1)+1)
                #fn_score = np.sum(fn_mask.reshape(fn_mask.shape[0],-1),1) \
                #    / (np.sum(origin_label_onehot[i].cpu().numpy().reshape(fn_mask.shape[0],-1),1)+1)
                bad_pred_class = fn_score >= cfg.INTERACTIVE.FN_THRESHOLD
                bad_pred_class_num = (bad_pred_class).sum()
                bad_pred_class_list += bad_pred_class
                bad_pred_class_index = np.where(fn_score >= cfg.INTERACTIVE.FN_THRESHOLD)[0]
                #print(fn_score, bad_pred_class_num, bad_pred_class_index)
                
                for j in range(bad_pred_class_num):
                    if active_pixels<1:
                        break
                    #bad_gt = origin_label_onehot[i][bad_pred_class_index[j]]
                    #bad_pred = one_hot[i][bad_pred_class_index[j]]
                    
                    fn_mask_dt = cv2.distanceTransform(fn_mask[bad_pred_class_index[j]].astype(np.uint8), cv2.DIST_L2, 0)
                    h, w = np.where(fn_mask_dt == np.max(fn_mask_dt))
                    h = h[0]
                    w = w[0]
                    # mask out
                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]
                    #输入才要cv2.circle(mask, (coords_y, coords_x), radius, 1, -1)
                    active_pixels_i -= 1

                for pixel in range(active_pixels_i):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])
    print(bad_pred_class_list)
    feature_extractor.train()
    classifier.train()

def PointSampling(cfg, feature_extractor, classifier, tgt_epoch_loader):
    feature_extractor.eval()
    classifier.eval()

    active_pixels = math.ceil(cfg.ACTIVE.PIXELS / len(cfg.ACTIVE.SELECT_ITER) / (1280 * 640) * (2048 * 1024))
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    mask_radius = cfg.ACTIVE.RADIUS_K

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):

                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                output = output.squeeze(dim=0)
                p = torch.softmax(output, dim=0)
                entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0)
                pseudo_label = torch.argmax(p, dim=0)
                one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
                one_hot = one_hot.permute((2, 0, 1)).unsqueeze(dim=0)
                purity = calculate_purity(one_hot).squeeze(dim=0).squeeze(dim=0)
                score = entropy * purity

                score[active] = -float('inf')

                for pixel in range(active_pixels):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    end_w = w + mask_radius + 1
                    end_h = h + mask_radius + 1
                    # mask out
                    score[start_h:end_h, start_w:end_w] = -float('inf')
                    active[start_h:end_h, start_w:end_w] = True
                    selected[h, w] = True
                    # active sampling
                    active_mask[h, w] = ground_truth[h, w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()

def RegionSelection(cfg, feature_extractor, classifier, tgt_epoch_loader):

    feature_extractor.eval()
    classifier.eval()

    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2
    active_ratio = cfg.ACTIVE.RATIO / len(cfg.ACTIVE.SELECT_ITER)

    with torch.no_grad():
        for tgt_data in tqdm(tgt_epoch_loader):

            tgt_input, path2mask = tgt_data['img'], tgt_data['path_to_mask']
            origin_mask, origin_label = \
                tgt_data['origin_mask'], tgt_data['origin_label']
            origin_size = tgt_data['size']
            active_indicator = tgt_data['active']
            selected_indicator = tgt_data['selected']
            path2indicator = tgt_data['path_to_indicator']

            tgt_input = tgt_input.cuda(non_blocking=True)

            tgt_size = tgt_input.shape[-2:]
            tgt_feat = feature_extractor(tgt_input)
            tgt_out = classifier(tgt_feat, size=tgt_size)

            for i in range(len(origin_mask)):
                active_mask = origin_mask[i].cuda(non_blocking=True)
                ground_truth = origin_label[i].cuda(non_blocking=True)
                size = (origin_size[i][0], origin_size[i][1])
                num_pixel_cur = size[0] * size[1]
                active = active_indicator[i]
                selected = selected_indicator[i]

                output = tgt_out[i:i + 1, :, :, :]
                output = F.interpolate(output, size=size, mode='bilinear', align_corners=True)
                score, purity, entropy = floating_region_score(output)

                score[active] = -float('inf')

                active_regions = math.ceil(num_pixel_cur * active_ratio / per_region_pixels)

                for pixel in range(active_regions):
                    values, indices_h = torch.max(score, dim=0)
                    _, indices_w = torch.max(values, dim=0)
                    w = indices_w.item()
                    h = indices_h[w].item()

                    active_start_w = w - active_radius if w - active_radius >= 0 else 0
                    active_start_h = h - active_radius if h - active_radius >= 0 else 0
                    active_end_w = w + active_radius + 1
                    active_end_h = h + active_radius + 1

                    mask_start_w = w - mask_radius if w - mask_radius >= 0 else 0
                    mask_start_h = h - mask_radius if h - mask_radius >= 0 else 0
                    mask_end_w = w + mask_radius + 1
                    mask_end_h = h + mask_radius + 1

                    # mask out
                    score[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = -float('inf')
                    active[mask_start_h:mask_end_h, mask_start_w:mask_end_w] = True
                    selected[active_start_h:active_end_h, active_start_w:active_end_w] = True
                    # active sampling
                    active_mask[active_start_h:active_end_h, active_start_w:active_end_w] = \
                        ground_truth[active_start_h:active_end_h, active_start_w:active_end_w]

                active_mask = Image.fromarray(np.array(active_mask.cpu().numpy(), dtype=np.uint8))
                active_mask.save(path2mask[i])
                indicator = {
                    'active': active,
                    'selected': selected
                }
                torch.save(indicator, path2indicator[i])

    feature_extractor.train()
    classifier.train()

