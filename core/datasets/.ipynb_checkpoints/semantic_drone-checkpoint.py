import os
import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
import pickle
import torch
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DroneDataSet(data.Dataset):
    def __init__(self,
                 data_root,
                 data_list,
                 max_iters=None,
                 num_classes=6,
                 split="train",
                 transform=None,
                 ignore_label=255,
                 debug=False,
                 cfg=None,
                 empty=False,
                 tgt=False):
        self.active = True if split == 'active' else False
        if split == 'active':
            split = 'train'

        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.cfg = cfg
        self.empty = empty
        self.tgt = tgt

        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]

        if self.tgt:
            if empty:
                self.data_list.append(
                    {
                        "img": "",
                        "label": "",
                        "label_mask": "",
                        "name": "",
                    }
                )
            else:
                for fname in content:
                    name = fname.strip().split('.')[0]
                    self.data_list.append(
                        {
                            "img": os.path.join(
                                self.data_root, "%s/images/%s" % (split, name + '.jpg')
                            ),
                            "label": os.path.join(
                                self.data_root,
                                "%s/label_trans/%s"
                                % (
                                    self.split,
                                    name + ".png",
                                ),
                            ),
                            "label_mask": os.path.join(
                                self.cfg.OUTPUT_DIR,
                                "gtMask/%s/%s"
                                % (
                                    self.split,
                                    name + "_gtFine_labelIds.png",
                                ),
                            ),
                            "name": name,
                            'indicator': os.path.join(
                                cfg.OUTPUT_DIR,
                                "gtIndicator/%s/%s"
                                % (
                                    "train",
                                    name + "_indicator.pth",
                                ),
                            )
                        }
                    )
        else:
            if max_iters is not None:
                self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, "vaihingen_label_info.p"), "rb"))
                self.img_ids = []
                SUB_EPOCH_SIZE = 3000
                tmp_list = []
                ind = dict()
                for i in range(self.NUM_CLASS):
                    ind[i] = 0
                for e in range(int(max_iters / SUB_EPOCH_SIZE) + 1):
                    cur_class_dist = np.zeros(self.NUM_CLASS)
                    for i in range(SUB_EPOCH_SIZE):
                        if cur_class_dist.sum() == 0:
                            dist1 = cur_class_dist.copy()
                        else:
                            dist1 = cur_class_dist / cur_class_dist.sum()
                        w = 1 / np.log(1 + 1e-2 + dist1)
                        w = w / w.sum()
                        c = np.random.choice(self.NUM_CLASS, p=w) # 随机选择某一个类别，优先选择没怎么见过的类别

                        if ind[c] > (len(self.label_to_file[c]) - 1): # 如果某个类别
                            np.random.shuffle(self.label_to_file[c])
                            ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)

                        c_file = self.label_to_file[c][ind[c]]
                        tmp_list.append(c_file)
                        ind[c] = ind[c] + 1 # label_to_file，每个类别排到第几个图片，总epoch中统计，用完打乱label_to_file，再取
                        cur_class_dist[self.file_to_label[c_file]] += 1 # file_to_label，统计所有已选图片中类别的分布情况，计算类别的选择概率

                self.img_ids = tmp_list

            for name in self.img_ids:
                self.data_list.append(
                    {
                        "img": os.path.join(self.data_root, "%s/images/%s" % (split, name.split('.')[0]+'.jpg')),
                        "label": os.path.join(self.data_root, "%s/label_trans/%s" % (split, name.split('.')[0]+'.png')),
                        "name": name,
                    }
                )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {29: 0,
                              76: 1,
                              149: 2,
                              178: 3,
                              225: 4,
                              255: 5}
        self.trainid2name = {
            0: "Building",
            1: "Clutter",
            2: "Tree",
            3: "Low Vegetation",
            4: "Car",
            5: "Road",
        }
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        #label = np.array(Image.open(datafiles["label"]), dtype=np.uint8)

        #label = cv2.imread(datafiles["labels"])

        label = cv2.imread(datafiles["label"], 0).astype(np.int32)
        if self.tgt:
            label_mask = None
            if self.split == 'train':
                    label_mask = np.array(Image.open(datafiles["label_mask"]), dtype=np.uint8)
            else:
                # test or val, mask is useless
                label_mask = np.ones_like(label, dtype=np.uint8) * 255
            
            # for generate new mask
            origin_mask = torch.from_numpy(label_mask).long()

            active_indicator = torch.tensor([0])
            active_selected = torch.tensor([0])

            indicator = torch.load(datafiles['indicator'])
            active_indicator = indicator['active']
            active_selected = indicator['selected']
            # if first time load, initialize it
            if active_indicator.size() == (1,):
                active_indicator = torch.zeros_like(origin_mask, dtype=torch.bool)
                active_selected = torch.zeros_like(origin_mask, dtype=torch.bool)

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        
        if self.tgt:
            label = np.array(label_copy, dtype=np.uint8)

            origin_label = torch.from_numpy(label).long()

            label.resize(label.shape[0], label.shape[1], 1)
            label_mask.resize(label_mask.shape[0], label_mask.shape[1], 1)

            h, w = label.shape[0], label.shape[1]

            mask_aggregation = np.concatenate((label, label_mask), axis=2)
            mask_aggregation = Image.fromarray(mask_aggregation)

            if self.transform is not None:
                image, mask_aggregation = self.transform(image, mask_aggregation)
                label = mask_aggregation[:, :, 0]
                label_mask = mask_aggregation[:, :, 1]

        else:
            label = Image.fromarray(label_copy)
            
            if self.transform is not None:
                image, label = self.transform(image, label)

        if self.tgt:
            ret_data = {
            "img": image,  # data
            'label': label,  # for test
            'mask': label_mask,  # for train
            'name': datafiles['name'],  # for test to store the results
            'path_to_mask': datafiles['label_mask'],  # for active to store new mask
            'path_to_indicator': datafiles['indicator'],  # store new indicator
            'size': torch.tensor([h, w]),  # for active to interpolate the output to original size
            'origin_mask': origin_mask,  # mask without transforms for active
            'origin_label': origin_label,  # label without transforms for active
            'active': active_indicator,  # indicate region or pixels can not be selected. 保证选到的位置要标注的3x3区域与已经标注的区域不重叠，因此半径是k的两倍
            'selected': active_selected,  # indicate the pixel have been selected, can calculate the class-wise ratio of selected samples
        }
        else:
            ret_data = {
                "img": image,
                'label': label,
                'name': datafiles['name'].split('.')[0],
                'index': index,
                'datafiles': datafiles,
            }

        return ret_data
