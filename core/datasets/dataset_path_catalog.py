import os
import os.path as op
from .semi import  SemiDataset
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from tqdm import tqdm
import errno


class DatasetCatalog(object):
    DATASET_DIR = "datasets"
    DATASETS = {
        "potsdam_train": {
            "data_dir": "potsdam",
            "data_list": "potsdam_train_list.txt",
        },
        "potsdam_test": {
            "data_dir": "potsdam",
            "data_list": "potsdam_test_list.txt",
        },
        "vaihingen_train": {
            "data_dir": "vaihingen",
            "data_list": "vaihingen_train_list.txt",
        },
        "vaihingen_test": {
            "data_dir": "vaihingen",
            "data_list": "vaihingen_test_list.txt",
        },
        "potsdam2_train": {
            "data_dir": "potsdam2",
            "data_list": "potsdam2_train_list.txt",
        },
        "drone_train": {
            "data_dir": "drone",
            "data_list": "drone_train_list.txt",
        },
        "drone_test": {
            "data_dir": "drone",
            "data_list": "drone_test_list.txt",
        }
    }

    @staticmethod
    def get(name, mode, num_classes, max_iters=None, transform=None, cfg=None, empty=False, tgt=False, size=None):
        data_dir = DatasetCatalog.DATASET_DIR
        attrs = DatasetCatalog.DATASETS[name]
        args = dict(
            root=os.path.join(data_dir, attrs["data_dir"]),
            data_list=os.path.join(data_dir, attrs["data_list"]),
        )
        return SemiDataset(args["root"], args["data_list"], size=size, max_iters=max_iters, num_classes=num_classes,
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

        data_list = os.path.join(data_dir, attrs["data_list"])
        root = os.path.join(data_dir, attrs["data_dir"])
        with open(data_list, "r") as handle:
            content = handle.readlines()
            
        mask_dir = os.path.join("%s/gtMask/train" % (cfg.OUTPUT_DIR))
        indicator_dir = os.path.join("%s/gtIndicator/train" % (cfg.OUTPUT_DIR))
        for fname in tqdm(content):
            name = fname.strip()

            path2image = os.path.join(root, "train/images/%s" % (name))
            path2mask = os.path.join(
                cfg.OUTPUT_DIR,
                "gtMask/%s/%s"
                % (
                    "train",
                    name.split('.')[0] + "_gtFine_labelIds.png",
                ),
            )
            path2indicator = os.path.join(
                cfg.OUTPUT_DIR,
                "gtIndicator/%s/%s"
                % (
                    "train",
                    name.split('.')[0] + "_indicator.pth",
                ),
            )


            # mkdir
            mkdir_path(mask_dir+'/'+name.split('/')[0])
            mkdir_path(indicator_dir+'/'+name.split('/')[0])

            img = Image.open(path2image.replace('\\', '/')).convert('RGB')
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
