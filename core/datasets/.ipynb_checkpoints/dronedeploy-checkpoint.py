import os
import os.path as osp
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
import pickle
import torch
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class potsdamDataSet(data.Dataset):
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

        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        self.tgt = tgt
        
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]

        if max_iters is not None:
            self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, "potsdam_label_info.p"), "rb"))
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
                    "img": os.path.join(self.data_root, "%s/images/%s" % (split, name)),
                    "label": os.path.join(self.data_root, "%s/labels/%s" % (split, name)),
                    "name": name.split('.')[0],
                }
            )

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {0: 108,
                              1: 70,
                              2: 119,
                              3: 90,
                              4: 16,
                              5: 0}
        self.trainid2name = {
            0: "wall",
            1: "roof",
            2: "vegetation",
            3: "road",
            4: "car",
            5: "other",
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

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        ret_data = {
            "img": image,
            'label': label,
            'index': index,
            'name': datafiles['name'],
            'datafiles': datafiles,
        }

        return ret_data
