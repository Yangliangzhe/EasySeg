from . import transform
from .dataset_path_catalog import DatasetCatalog
from albumentations.augmentations.geometric.transforms import HorizontalFlip, PadIfNeeded
from albumentations.augmentations.transforms import RandomBrightnessContrast, RGBShift
from albumentations import ImageOnlyTransform, DualTransform
import cv2
import random
from albumentations.augmentations.geometric import functional as F

class UniformRandomResize(DualTransform):
    def __init__(self, scale_range=(0.9, 1.1), interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.interpolation = interpolation

    def get_params_dependent_on_targets(self, params):
        scale = random.uniform(*self.scale_range)
        height = int(round(params['image'].shape[0] * scale))
        width = int(round(params['image'].shape[1] * scale))
        return {'new_height': height, 'new_width': width}

    def apply(self, img, new_height=0, new_width=0, interpolation=cv2.INTER_LINEAR, **params):
        return F.resize(img, height=new_height, width=new_width, interpolation=interpolation)

    def apply_to_keypoint(self, keypoint, new_height=0, new_width=0, **params):
        scale_x = new_width / params["cols"]
        scale_y = new_height / params["rows"]
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return "scale_range", "interpolation"

    @property
    def targets_as_params(self):
        return ["image"]


def build_transform(cfg, mode, is_source):
    if mode == "train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            #UniformRandomResize(scale_range=(0.75, 1.40)),
            transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
            #transform.HorizontalFlip(),
            #PadIfNeeded(min_height=h, min_width=w, border_mode=0),
            transform.RandomCrop(size=(h, w), pad_if_needed=True),
            #transform.RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
            #transform.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255),
        ]
        #if cfg.INPUT.INPUT_SCALES_TRAIN[0] == cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0] == 1:
        #    trans_list = [transform.Resize((h, w)), ] + trans_list
        #else:
        #    trans_list = [
        #                     transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
        #                     transform.RandomCrop(size=(h, w), pad_if_needed=True),
        #                 ] + trans_list
        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans


def build_dataset(cfg, mode='train', is_source=True, epochwise=False, empty=False):
    assert mode in ['train', 'val', 'test', 'active']
    transform = build_transform(cfg, mode, is_source)
    print('!!!transform!!!', transform)
    iters = None
    if mode == 'train' or mode == 'active':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER * cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg, empty=empty)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES,
                                         max_iters=iters, transform=transform, cfg=cfg, empty=empty, tgt=True)
    elif mode == 'val':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, 'val', num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters,
                                     transform=transform, cfg=cfg, empty=empty)
    elif mode == 'test':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, cfg.DATASETS.TEST.split('_')[-1],
                                     num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform, cfg=cfg,
                                     empty=empty)

    return dataset
