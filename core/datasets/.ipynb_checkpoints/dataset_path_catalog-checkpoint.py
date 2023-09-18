import os
import os.path as op
from .cityscapes import cityscapesDataSet
from .gtav import GTAVDataSet
from .synthia import synthiaDataSet
from .potsdam import potsdamDataSet
from .vaihingen import vaihingenDataSet
from .semantic_drone import DroneDataSet
from .UDD6 import UDD6DataSet
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import errno


class DatasetCatalog(object):
    DATASET_DIR = "/root/autodl-tmp/AdaptSegNet-master/dataset/"
    DATASETS = {
        "gtav_train": {
            "data_dir": "gtav",
            "data_list": "gtav_train_list.txt"
        },
        "synthia_train": {
            "data_dir": "synthia",
            "data_list": "synthia_train_list.txt"
        },
        "cityscapes_train": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_train_list.txt"
        },
        "cityscapes_val": {
            "data_dir": "cityscapes",
            "data_list": "cityscapes_val_list.txt"
        },
        "potsdam_train": {
            "data_dir": "potsdam",
            "data_list": "potsdam_train_list.txt",
        },
        "potsdam_test": {
            "data_dir": "potsdam",
            "data_list": "potsdam_test_list.txt",
        },
        "potsdam2_train": {
            "data_dir": "potsdam2",
            "data_list": "potsdam2_train_list.txt",
        },
        "vaihingen_train": {
            "data_dir": "vaihingen",
            "data_list": "vaihingen_train_list.txt",
        },
        "vaihingen_test": {
            "data_dir": "vaihingen",
            "data_list": "vaihingen_test_list.txt",
        },
        "drone_train": {
            "data_dir": "drone",
            "data_list": "drone_train_list.txt",
        },
        "drone_test": {
            "data_dir": "drone",
            "data_list": "drone_test_list.txt",
        },
        "udd6_train": {
            "data_dir": "udd6",
            "data_list": "udd6_train_list.txt",
        },
        "udd6_test": {
            "data_dir": "udd6",
            "data_list": "udd6_test_list.txt",
        },

    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, cfg=None, empty=False, tgt=False):
        if "gtav" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return GTAVDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                               split=mode, transform=transform)
        elif "synthia" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return synthiaDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                  split=mode, transform=transform)

        elif "cityscapes" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return cityscapesDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty)
        
        elif "vaihingen" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return vaihingenDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty, tgt=tgt)
        elif "udd6" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return UDD6DataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty, tgt=tgt)
        
        elif "drone" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return DroneDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty, tgt=tgt)
        
        elif "potsdam" in name:
            data_dir = DatasetCatalog.DATASET_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["data_dir"]),
                data_list=os.path.join(data_dir, attrs["data_list"]),
            )
            return potsdamDataSet(args["root"], args["data_list"], max_iters=max_iters, num_classes=num_classes,
                                     split=mode, transform=transform, cfg=cfg, empty=empty, tgt=tgt)

        raise RuntimeError("Dataset not available: {}".format(name))

    @staticmethod
    def initMask(cfg):
        if cfg.DEBUG == 1:
            for i in range(10):
                print("Debug without mask initialization!")
            return
        data_dir = DatasetCatalog.DATASET_DIR
        attrs = DatasetCatalog.DATASETS[cfg.DATASETS.TARGET_TRAIN]

        #attrs = DatasetCatalog.DATASETS['cityscapes_train']
        data_list = os.path.join(data_dir, attrs["data_list"])
        root = os.path.join(data_dir, attrs["data_dir"])
        with open(data_list, "r") as handle:
            content = handle.readlines()
            
        mask_dir = os.path.join("%s/gtMask/train" % (cfg.OUTPUT_DIR))
        indicator_dir = os.path.join("%s/gtIndicator/train" % (cfg.OUTPUT_DIR))
        for fname in tqdm(content):
            name = fname.strip().split('.')[0]

            path2image = os.path.join(root, "train/images/%s" % (fname.strip()))
            #path2image = os.path.join(root, "leftImg8bit/%s/%s" % ('train', name))
            path2mask = os.path.join(
                cfg.OUTPUT_DIR,
                "gtMask/%s/%s"
                % (
                    "train",
                    name + "_gtFine_labelIds.png",
                ),
            )
            path2indicator = os.path.join(
                cfg.OUTPUT_DIR,
                "gtIndicator/%s/%s"
                % (
                    "train",
                    name + "_indicator.pth",
                ),
            )


            # mkdir
            mkdir_path(mask_dir)
            mkdir_path(indicator_dir)

            img = Image.open(path2image).convert('RGB')
            h, w = img.size[1], img.size[0]
            mask = np.ones((h, w), dtype=np.uint8) * 255
            mask = Image.fromarray(mask)
            mask.save(path2mask)

            indicator = {
                'active': torch.tensor([0], dtype=torch.bool),
                'selected': torch.tensor([0], dtype=torch.bool),
            }
            torch.save(indicator, path2indicator)


def mkdir_path(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
