import argparse
import os
import datetime
import logging
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils
import torch.distributed
from torch.utils.data import DataLoader
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, get_color_pallete
from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.active.build import RIPUPixelSelection, SFALPixelSelection, runningScore
from core.datasets.dataset_path_catalog import DatasetCatalog
from core.loss.negative_learning_loss import NegativeLearningLoss
from core.loss.local_consistent_loss import LocalConsistentLoss
from core.utils.misc import mkdir, get_color_pallete
from core.active.spatial_purity import SpatialPurity
from core.utils.utils import set_random_seed
import torch.backends.cudnn as cudnn
from core.utils.dist_helper import setup_distributed
import random
import numpy as np
import cv2

import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')


def train(cfg, args):
    logger = logging.getLogger("EasySeg.trainer")

    # create network
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor = build_feature_extractor(cfg)
    classifier = build_classifier(cfg)

    rank, world_size = setup_distributed(port=args.port)
    print('nGPU: ', world_size)
    cudnn.enabled = True
    cudnn.benchmark = True
    local_rank = int(os.environ["LOCAL_RANK"])
    feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
    classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier.cuda()
    feature_extractor.cuda()
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[local_rank], broadcast_buffers=False,
                                                           output_device=local_rank, find_unused_parameters=True)


    exp_name = ['maps_transform']
    params = []
    params2 = []

    for name, param in feature_extractor.named_parameters():
        param_group = {'params': [param]}
        param_group2 = {'params': [param]}
        for i in exp_name:
            if i in name:
                param_group['lr'] = cfg.SOLVER.BASE_LR * 10
                params.append(param_group)
            else:
                param_group2['lr'] = cfg.SOLVER.BASE_LR
                params2.append(param_group2)

    # init optimizer
    optimizer_fea = torch.optim.SGD(params2, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    optimizer_inter = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_inter.zero_grad()
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR * 10, momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()

    # init mask for target domain
    DatasetCatalog.initMask(cfg)

    # init data loader
    src_train_data = build_dataset(cfg, mode='train', is_source=True, size=cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False, size=cfg.INPUT.TARGET_INPUT_SIZE_TRAIN)
    tgt_epoch_data = build_dataset(cfg, mode='active', is_source=False, epochwise=True, size=cfg.INPUT.INPUT_SIZE_TEST)
    test_data = build_dataset(cfg, mode='test', is_source=False, size=cfg.INPUT.INPUT_SIZE_TEST)

    trainsampler_src = torch.utils.data.distributed.DistributedSampler(src_train_data)
    src_train_loader = DataLoader(src_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=1\
                                  , drop_last=True, sampler=trainsampler_src)
    trainsampler_tgt = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    tgt_train_loader = DataLoader(tgt_train_data, batch_size=cfg.SOLVER.BATCH_SIZE, num_workers=1\
                                  , drop_last=True, sampler=trainsampler_tgt)
    activesampler = torch.utils.data.distributed.DistributedSampler(tgt_epoch_data)
    tgt_epoch_loader = DataLoader(tgt_epoch_data, batch_size=1, shuffle=False, num_workers=1\
                                  , drop_last=False, sampler=activesampler)
    testsampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_loader = DataLoader(test_data, batch_size=1, num_workers=1,\
                             drop_last=False, sampler=testsampler)

    # init loss
    sup_criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    clickloss_criterion = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
    negative_criterion = NegativeLearningLoss(threshold=cfg.SOLVER.NEGATIVE_THRESHOLD).cuda(local_rank)
    local_consistent_loss = LocalConsistentLoss(cfg.MODEL.NUM_CLASSES, cfg.SOLVER.LCR_TYPE).cuda(local_rank)
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    # action function
    calculate_purity = SpatialPurity(in_channels=6, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda(local_rank)

    iteration = 1
    iteration_thre = 1500 # mask warm-up iterations
    previous_best = 0.0
    start_training_time = time.time()
    end = time.time()
    max_iters = cfg.SOLVER.MAX_ITER
    meters = MetricLogger(delimiter="  ")

    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(cfg.MODEL.NUM_CLASSES):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # load checkpoint
    if cfg.resume:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        classifier.load_state_dict(checkpoint['classifier'])
        optimizer_fea.load_state_dict(checkpoint['optimizer_fea'])
        optimizer_inter.load_state_dict(checkpoint['optimizer_inter'])
        optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
        iteration = checkpoint['iteration']
        previous_best = checkpoint['previous_best']
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % iteration)

    logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
    feature_extractor.train()
    classifier.train()
    output_folder = os.path.join(cfg.OUTPUT_DIR, "train", cfg.DATASETS.SOURCE_TRAIN)
    mkdir(output_folder)

    active_round = 1
    loader = zip(src_train_loader, tgt_train_loader, tgt_train_loader)
    for batch_index, (src_data, tgt_data1, tgt_data2) in enumerate(loader):
        img_src, label_src = src_data["img"], src_data["label"]

        img_tgt_w, img_tgt_s1, img_tgt_s2, ignore_mask, cutmix_box1, cutmix_box2, tgt_mask1 = \
            tgt_data1["img"], tgt_data1["img_s1"], tgt_data1["img_s2"], tgt_data1["ignore_mask"], tgt_data1["cutmix_box1"], tgt_data1["cutmix_box2"], tgt_data1["mask"]
        img_tgt_w_mix, img_tgt_s1_mix, img_tgt_s2_mix, ignore_mask_mix, tgt_mask2 = \
            tgt_data2["img"], tgt_data2["img_s1"], tgt_data2["img_s2"], tgt_data2["ignore_mask"], tgt_data2["mask"]
        data_time = time.time() - end

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters,
                                          power=cfg.SOLVER.LR_POWER)

        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_inter.param_groups)):
            if iteration >= cfg.INTERACTIVE.WARMUP_ITER:
                optimizer_inter.param_groups[index]['lr'] = current_lr  # / 10
            else:
                optimizer_inter.param_groups[index]['lr'] = current_lr * 10
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr * 10

        img_src, label_src = img_src.cuda(), label_src.cuda()
        img_tgt_w = img_tgt_w.cuda()
        img_tgt_s1, img_tgt_s2, ignore_mask = img_tgt_s1.cuda(), img_tgt_s2.cuda(), ignore_mask.cuda()
        cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
        img_tgt_w_mix = img_tgt_w_mix.cuda()
        img_tgt_s1_mix, img_tgt_s2_mix = img_tgt_s1_mix.cuda(), img_tgt_s2_mix.cuda()
        ignore_mask_mix = ignore_mask_mix.cuda()
        tgt_mask1 = tgt_mask1.cuda()

        size = img_src.shape[-2:]

        if torch.sum((tgt_mask2 != 255)) != 0:
            with torch.no_grad():
                feature_extractor.eval()
                classifier.eval()
                points = F.one_hot(torch.clamp(tgt_mask2, 0, 6), 7).permute(0, 3, 1, 2)[:, 0:6].cuda()
                kernel = torch.FloatTensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], \
                                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], \
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], \
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], \
                                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]).expand(points.shape[1], 1, 11, 11)
                weight = nn.Parameter(data=kernel, requires_grad=False).cuda()
                encoding_points = nn.functional.conv2d(points.float(), weight, bias=None, stride=1, padding=5, groups=6)
                prev_out = torch.zeros(encoding_points.shape).cuda()
                interactive_maps = torch.cat([encoding_points, prev_out], 1)

                pred_tgt_w_mix = classifier(feature_extractor(img_tgt_w_mix, interactive_maps), size=size).detach()

            pred_tgt_iseg_mix = pred_tgt_w_mix.softmax(dim=1)
            entropy_tgt_iseg_mix = torch.sum(-pred_tgt_iseg_mix * torch.log(pred_tgt_iseg_mix + 1e-6), dim=1)
            label_tgt_w_mix = pred_tgt_w_mix.argmax(dim=1)

        img_tgt_s1[cutmix_box1.unsqueeze(1).expand(img_tgt_s1.shape) == 1] = \
            img_tgt_s1_mix[cutmix_box1.unsqueeze(1).expand(img_tgt_s1.shape) == 1]
        img_tgt_s2[cutmix_box2.unsqueeze(1).expand(img_tgt_s2.shape) == 1] = \
            img_tgt_s2_mix[cutmix_box2.unsqueeze(1).expand(img_tgt_s2.shape) == 1]

        feature_extractor.train()
        classifier.train()

        num_lb, num_ulb = img_src.shape[0], img_tgt_w.shape[0]
        if torch.sum((tgt_mask1 != 255)) == 0:
            pred_src = classifier(feature_extractor(torch.cat((img_src, img_tgt_w))), size=size)
            pred_src, pred_tgt_w = pred_src.split([num_lb, num_ulb])
        else:
            preds, preds_fp = classifier(feature_extractor(torch.cat((img_src, img_tgt_w))), size=size, need_fp=True)
            pred_src, pred_tgt_w = preds.split([num_lb, num_ulb])
            pred_tgt_w_fp = preds_fp[num_lb:]

        loss = torch.Tensor([0]).cuda()
        loss_seg_sup_src = sup_criterion(pred_src, label_src) / 2.0
        loss += loss_seg_sup_src
        meters.update(loss_seg_sup_src=loss_seg_sup_src.item())

        src_label_onehot = F.one_hot(torch.clamp(label_src, 0, 6), num_classes=7).permute(0, 3, 1, 2)[:,0:6].cpu().numpy()
        seg_iou = get_iou(src_label_onehot, torch.softmax(pred_src, dim=1).detach().cpu().numpy())

        # Semi-supervised Learning
        if torch.sum((tgt_mask1 != 255)) != 0:  # target domain has labeled pixels
            pred_tgt_s1, pred_tgt_s2 = classifier(feature_extractor(torch.cat((img_tgt_s1, img_tgt_s2))), size=size).chunk(2)

            # Pseudo-labels Generation
            points = F.one_hot(torch.clamp(tgt_mask1, 0, 6), 7).permute(0, 3, 1, 2)[:, 0:6].cuda()
            kernel = torch.FloatTensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], \
                                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], \
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], \
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], \
                                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], \
                                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]).expand(points.shape[1], 1, 11, 11)
            weight = nn.Parameter(data=kernel, requires_grad=False).cuda()
            encoding_points = nn.functional.conv2d(points.float(), weight, bias=None, stride=1, padding=5, groups=6)
            interactive_maps = torch.cat([encoding_points, pred_tgt_w.softmax(dim=1).clone().detach()], 1)

            pred_tgt_iseg = classifier(feature_extractor(img_tgt_w, interactive_maps), size=size)
            pred_tgt_iseg_ = pred_tgt_iseg.softmax(dim=1).detach()
            entropy_tgt_iseg = torch.sum(-pred_tgt_iseg_ * torch.log(pred_tgt_iseg_ + 1e-6), dim=1)
            label_tgt_iseg = pred_tgt_iseg.argmax(dim=1).detach()

            loss_seg_semi_sup_tgt = criterion_u(pred_tgt_w, label_tgt_iseg) * ((entropy_tgt_iseg < cfg.INTERACTIVE.POSITIVE_THRESHOLD) \
                                                    & (ignore_mask != 255))
            loss_seg_semi_sup_tgt = loss_seg_semi_sup_tgt.sum() * cfg.INTERACTIVE.POSITIVE_LOSS / (1+(((entropy_tgt_iseg < cfg.INTERACTIVE.POSITIVE_THRESHOLD) \
                                                    & (ignore_mask != 255))).sum().item())
            loss_click_loss_tgt = clickloss_criterion(pred_tgt_iseg, tgt_mask1) * cfg.INTERACTIVE.CLICK_LOSS
            loss_seg_sup_tgt = sup_criterion(pred_tgt_w, tgt_mask1)

            meters.update(loss_seg_sup_tgt=loss_seg_sup_tgt.item() / 2.0, loss_seg_semi_sup_tgt=loss_seg_semi_sup_tgt.item() / 2.0,
                          loss_click_loss_tgt=loss_click_loss_tgt.item() / 2.0)
            loss += (loss_seg_sup_tgt + loss_seg_semi_sup_tgt + loss_click_loss_tgt) / 2.0

            # Weak-To-Strong Consistency
            label_tgt_w_cutmixed1, entropy_tgt_iseg_cutmixed1, ignore_mask_cutmixed1 = \
                label_tgt_iseg.clone(), entropy_tgt_iseg.clone(), ignore_mask.clone()
            label_tgt_w_cutmixed2, entropy_tgt_iseg_cutmixed2, ignore_mask_cutmixed2 = \
                label_tgt_iseg.clone(), entropy_tgt_iseg.clone(), ignore_mask.clone()

            label_tgt_w_cutmixed1[cutmix_box1 == 1] = label_tgt_w_mix[cutmix_box1 == 1]
            entropy_tgt_iseg_cutmixed1[cutmix_box1 == 1] = entropy_tgt_iseg_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            label_tgt_w_cutmixed2[cutmix_box2 == 1] = label_tgt_w_mix[cutmix_box2 == 1]
            entropy_tgt_iseg_cutmixed2[cutmix_box2 == 1] = entropy_tgt_iseg_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_tgt_s1 = criterion_u(pred_tgt_s1, label_tgt_w_cutmixed1)
            loss_tgt_s1 = loss_tgt_s1 * ((entropy_tgt_iseg_cutmixed1 < cfg.INTERACTIVE.POSITIVE_THRESHOLD) & (ignore_mask_cutmixed1 != 255))
            loss_tgt_s1 = loss_tgt_s1.sum() / (1+((entropy_tgt_iseg_cutmixed1 < cfg.INTERACTIVE.POSITIVE_THRESHOLD) & (ignore_mask_cutmixed1 != 255)).sum().item())

            loss_tgt_s2 = criterion_u(pred_tgt_s2, label_tgt_w_cutmixed2)
            loss_tgt_s2 = loss_tgt_s2 * ((entropy_tgt_iseg_cutmixed2 < cfg.INTERACTIVE.POSITIVE_THRESHOLD) & (ignore_mask_cutmixed2 != 255))
            loss_tgt_s2 = loss_tgt_s2.sum() / (1+((entropy_tgt_iseg_cutmixed2 < cfg.INTERACTIVE.POSITIVE_THRESHOLD) & (ignore_mask_cutmixed2 != 255)).sum().item())

            loss_tgt_w_fp = criterion_u(pred_tgt_w_fp, label_tgt_iseg)
            loss_tgt_w_fp = loss_tgt_w_fp * ((entropy_tgt_iseg < cfg.INTERACTIVE.POSITIVE_THRESHOLD) & (ignore_mask != 255))
            loss_tgt_w_fp = loss_tgt_w_fp.sum() / (1+((entropy_tgt_iseg < cfg.INTERACTIVE.POSITIVE_THRESHOLD) & (ignore_mask != 255)).sum().item())

            loss += (loss_tgt_s1 * 0.5 + loss_tgt_s2 * 0.5 + loss_tgt_w_fp) / 2.0
            label_ratio = (((entropy_tgt_iseg < cfg.INTERACTIVE.POSITIVE_THRESHOLD)) & (ignore_mask != 255)).sum().item() / \
                          (ignore_mask != 255).sum()
            meters.update(loss_tgt_s=(loss_tgt_s1.item() + loss_tgt_s2.item()) / 4.0, \
                          loss_tgt_fp=loss_tgt_w_fp.item() / 2.0, label_ratio=label_ratio)


        if cfg.SOLVER.CONSISTENT_LOSS > 0:
            consistent_loss_seg = local_consistent_loss(pred_src, label_src) * cfg.SOLVER.CONSISTENT_LOSS
            meters.update(cr_loss_seg=consistent_loss_seg.item() * cfg.SOLVER.CONSISTENT_LOSS)
            loss += consistent_loss_seg

        # target negative pseudo loss
        if cfg.SOLVER.NEGATIVE_LOSS > 0:
            tgt_predict = torch.softmax(pred_tgt_w, dim=1)
            negative_learning_loss_seg = negative_criterion(tgt_predict) * cfg.SOLVER.NEGATIVE_LOSS
            meters.update(nl_loss_seg=negative_learning_loss_seg.item() * cfg.SOLVER.NEGATIVE_LOSS)
            loss += negative_learning_loss_seg

        num_iters = random.randint(1, cfg.INTERACTIVE.max_num_next_clicks)  # 最大20，但最多19个点
        points = [[[-1, -1, -1] for _ in range(cfg.INTERACTIVE.max_num_next_clicks)] \
                  for _ in range(cfg.SOLVER.BATCH_SIZE)]

        # Interactive Semantic Segmentation Learning
        with torch.no_grad():
            feature_extractor.eval()
            classifier.eval()
            neediter = num_iters // 8
            haveplace = 0
            for i in range(neediter):
                if i + 1 == neediter:
                    num = num_iters % 8
                else:
                    num = 8

                prev_out = torch.softmax(pred_src, dim=1).clone().detach()
                points, interactive_maps, _ = place_next_points(prev_out, label_src, points, haveplace, iteration, \
                                                                calculate_purity, iteration_thre, radius=5,
                                                                num=num, \
                                                                FN_THRESHOLD=cfg.INTERACTIVE.FN_THRESHOLD, \
                                                                RADIUS_K=cfg.ACTIVE.RADIUS_K)

                pred_src = classifier(feature_extractor(img_src, interactive_maps), size=size)
                haveplace += num

        feature_extractor.train()
        classifier.train()
        prev_out = torch.softmax(pred_src, dim=1).clone().detach()
        points, interactive_maps, click_maps = place_next_points(prev_out, label_src, points, num_iters - 1, iteration,
                                                                 calculate_purity,
                                                                 iteration_thre, radius=5, num=1,
                                                                 FN_THRESHOLD=cfg.INTERACTIVE.FN_THRESHOLD, \
                                                                 RADIUS_K=cfg.ACTIVE.RADIUS_K)
        iseg_feat = feature_extractor(img_src, interactive_maps)
        src_out = classifier(iseg_feat, size=size)

        loss_iseg_sup = sup_criterion(src_out, label_src)
        click_maps = click_maps.clone().detach()
        click_maps = click_maps.argmax(1) + (click_maps.sum(1) == 0) * 255
        clickloss = clickloss_criterion(src_out, click_maps)

        iseg_iou = get_iou(src_label_onehot, torch.softmax(src_out, dim=1).detach().cpu().numpy())


        if iteration > cfg.INTERACTIVE.WARMUP_ITER:
            meters.update(loss_iseg_sup=loss_iseg_sup.item() * cfg.INTERACTIVE.ISEGLOSS_DOWN / 2.0)
            loss += loss_iseg_sup * cfg.INTERACTIVE.ISEGLOSS_DOWN / 2.0

            meters.update(clickloss=clickloss.item() * cfg.INTERACTIVE.ISEGLOSS_DOWN / 2.0)
            loss += clickloss * cfg.INTERACTIVE.ISEGLOSS_DOWN / 2.0

        else:
            meters.update(loss_iseg_sup=loss_iseg_sup.item() / 2.0)
            loss += loss_iseg_sup / 2.0

            meters.update(clickloss=clickloss.item() * cfg.INTERACTIVE.CLICK_LOSS / 2.0)
            loss += clickloss * cfg.INTERACTIVE.CLICK_LOSS / 2.0


        torch.distributed.barrier()
        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_inter.zero_grad()
        loss.backward()
        optimizer_fea.step()
        optimizer_inter.step()
        optimizer_cls.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "seg_iou: {seg_iou:.04f}",
                        "iseg_iou: {iseg_iou:.04f}",
                        "max mem: {memory:.02f} GB"
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer_fea.param_groups[0]["lr"],
                    seg_iou=seg_iou,
                    iseg_iou=iseg_iou,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
                )
            )

        if iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            filename = os.path.join(cfg.OUTPUT_DIR, "model_iter{:06d}.pth".format(iteration))
            torch.save({
                    'feature_extractor': feature_extractor.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer_fea': optimizer_fea.state_dict(),
                    'optimizer_inter': optimizer_inter.state_dict(),
                    'optimizer_cls': optimizer_cls.state_dict(),
                    'iteration': iteration,
                    'previous_best': previous_best,
                }, filename)

        # Image Labeling
        if iteration in cfg.ACTIVE.SELECT_ITER or cfg.DEBUG:
            if cfg.INTERACTIVE.select == 'SFAL':
                SFALPixelSelection(cfg=cfg,
                                    feature_extractor=feature_extractor,
                                    classifier=classifier,
                                    tgt_epoch_loader=tgt_epoch_loader,
                                    cuda_id=device)
            elif cfg.INTERACTIVE.select == 'RIPU':
                RIPUPixelSelection(cfg=cfg,
                                    feature_extractor=feature_extractor,
                                    classifier=classifier,
                                    tgt_epoch_loader=tgt_epoch_loader,
                                    cuda_id=device)
            else:
                print('Please choose labeling strategy in [\'SFAL\', \'RIPU\'].')
            active_round += 1

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
        iteration += 1
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / cfg.SOLVER.STOP_ITER
        )
    )


def place_next_points(pred, gt, points, click_indx, iteration, calculate_purity, iteration_thre=100, pred_thresh=0.50,
                      radius=5, num=1, FN_THRESHOLD=0.3, RADIUS_K=3):
    assert click_indx >= 0
    B, C, H, W = pred.shape
    #RADIUS_K = 50
    scorecal = runningScore(C)
    gt_onehot = F.one_hot(torch.clamp(gt, 0, 6), num_classes=C+1).permute(0, 3, 1, 2)[:,0:6].cpu().numpy()
    pred_label = pred.argmax(1)
    pred_onehot = F.one_hot(pred_label, num_classes=pred.shape[1]).permute(0, 3, 1, 2).detach().cpu().numpy()
    fn_mask = np.logical_and(gt_onehot, np.logical_not(pred_onehot)).astype(np.uint8)

    fn_mask = fn_mask.reshape(B * C, H, W)
    fn_mask_dt_list = np.zeros(fn_mask.shape)
    for bcindx in range(fn_mask.shape[0]):
        fn_mask_dt_list[bcindx] = cv2.distanceTransform(fn_mask[bcindx], cv2.DIST_L2, 0)  # [1:-1, 1:-1]

    fn_mask_dt_list = fn_mask_dt_list.reshape(B, C, H, W)
    interactive_maps = torch.zeros(pred.shape).to(pred.device)
    click_maps = torch.zeros(pred.shape).to(pred.device)

    bnum = np.ones((B,)) * num
    for bindx in range(B):
        neednum = int(bnum[bindx])
        haveplace = 0
        lscore = scorecal.get_labelscores(gt[bindx].cpu().numpy(), pred_label[bindx].cpu().numpy())
        bad_pred_class_num = (lscore >= FN_THRESHOLD).sum()
        bad_pred_class_index = np.where(lscore >= FN_THRESHOLD)[0]
        for j in range(bad_pred_class_num):
            if neednum < 1:
                break
            dt = fn_mask_dt_list[bindx, bad_pred_class_index[j]]
            max_dt = np.max(dt)
            if max_dt > 5:
                h, w = np.where(dt == max_dt)
                h = h[0]
                w = w[0]
                points[bindx][click_indx + haveplace][0] = h
                points[bindx][click_indx + haveplace][1] = w
                points[bindx][click_indx + haveplace][2] = bad_pred_class_index[j]
                neednum -= 1
                haveplace += 1

        if neednum >= 1:
            entropy = torch.sum(-pred[bindx] * torch.log(pred[bindx] + 1e-6), dim=0)
            purity = calculate_purity(pred[bindx].unsqueeze(dim=0)).squeeze(dim=0).squeeze(dim=0)
            score = entropy * purity
            for clk in range(click_indx + haveplace):
                h, w, _ = points[bindx][clk]
                start_w = w - RADIUS_K if w - RADIUS_K >= 0 else 0
                start_h = h - RADIUS_K if h - RADIUS_K >= 0 else 0
                end_w = w + RADIUS_K + 1
                end_h = h + RADIUS_K + 1

                score[start_h:end_h, start_w:end_w] = -float('inf')

        for _ in range(neednum):
            values, indices_h = torch.max(score, dim=0)
            _, indices_w = torch.max(values, dim=0)
            w = indices_w.item()
            h = indices_h[w].item()
            
            points[bindx][click_indx + haveplace][0] = h
            points[bindx][click_indx + haveplace][1] = w
            points[bindx][click_indx + haveplace][2] = np.argmax(gt_onehot[bindx, :, h, w])

            start_w = w - RADIUS_K if w - RADIUS_K >= 0 else 0
            start_h = h - RADIUS_K if h - RADIUS_K >= 0 else 0
            end_w = w + RADIUS_K + 1
            end_h = h + RADIUS_K + 1

            haveplace += 1
            score[start_h:end_h, start_w:end_w] = -float('inf')

        assert haveplace == bnum[bindx]
        for indx in range(click_indx + haveplace):
            click_map = np.zeros((H, W), dtype=np.uint8)
            
            interactive_maps[bindx, int(points[bindx][indx][2])] = \
                torch.max(interactive_maps[bindx, int(points[bindx][indx][2])], \
                          torch.tensor(
                              cv2.circle(click_map, (int(points[bindx][indx][1]), int(points[bindx][indx][0])), radius,
                                         1, -1)).to(pred.device))
            click_maps[bindx, int(points[bindx][indx][2]), int(points[bindx][indx][0]), int(points[bindx][indx][1])] = 1

    if iteration > iteration_thre:
        interactive_maps = torch.cat([interactive_maps, pred.clone().detach()], 1)
    else:
        interactive_maps = torch.cat(
            [interactive_maps, torch.zeros(interactive_maps.shape).to(interactive_maps.device)], 1)

    return points, interactive_maps, click_maps


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    pred_mask = pred_mask.argmax(1)
    pred_mask = F.one_hot(torch.tensor(pred_mask), 6).permute(0, 3, 1, 2).numpy()
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def main():
    parser = argparse.ArgumentParser(description="Active Domain Adaptive Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    parser.add_argument("--proctitle",
                        type=str,
                        default="EasySeg",
                        help="allow a process to change its title", )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)

    args = parser.parse_args()

    if args.opts is not None:
        args.opts[-1] = args.opts[-1].strip('\r\n')

    torch.backends.cudnn.benchmark = True

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("EASE-ADA", output_dir, 0)
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    logger.info('Initializing Target Domain label mask...')

    set_random_seed(cfg.SEED)

    train(cfg,args)


if __name__ == '__main__':
    main()
