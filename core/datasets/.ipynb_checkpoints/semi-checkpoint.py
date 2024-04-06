import os
import os.path as osp
from .transform import *
import numpy as np
from torch.utils import data
from copy import deepcopy
from PIL import Image, ImageFile
import pickle
from torchvision import transforms
import random
import torch
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SemiDataset(data.Dataset):
    def __init__(self,
                 data_root,
                 data_list,
                 size=None,
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
        self.size= size
        self.prob = 0
        self.src_aug = cfg.WTS_ADA.SRC_AUG

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
                    name = fname.strip()
                    self.data_list.append(
                        {
                            "img": os.path.join(
                                self.data_root, "%s/images/%s" % (split, name)
                            ),
                            "label": os.path.join(
                                self.data_root,
                                "%s/labels/%s"
                                % (
                                    self.split,
                                    name.split('.')[0] + ".png",
                                ),
                            ),
                            "label_mask": os.path.join(
                                self.cfg.OUTPUT_DIR,
                                "gtMask/%s/%s"
                                % (
                                    self.split,
                                    name.split('.')[0] + "_gtFine_labelIds.png",
                                ),
                            ),
                            "name": name,
                            'indicator': os.path.join(
                                cfg.OUTPUT_DIR,
                                "gtIndicator/%s/%s"
                                % (
                                    "train",
                                    name.split('.')[0] + "_indicator.pth",
                                ),
                            )
                        }
                    )
        else:
            if max_iters is not None:
                if self.tgt:
                    info = "vaihingen_label_info.p" # "drone_label_info.p"
                else:
                    info = "potsdam_label_info.p"
                self.label_to_file, self.file_to_label = pickle.load(
                    open(osp.join(data_root, info), "rb"))
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
                        c = np.random.choice(self.NUM_CLASS, p=w)

                        if ind[c] > (len(self.label_to_file[c]) - 1):
                            np.random.shuffle(self.label_to_file[c])
                            ind[c] = ind[c] % (len(self.label_to_file[c]) - 1)

                        c_file = self.label_to_file[c][ind[c]]
                        tmp_list.append(c_file)
                        ind[c] = ind[c] + 1  

                        cur_class_dist[self.file_to_label[c_file]] += 1

                self.img_ids = tmp_list

            for name in self.img_ids:
                name = name.strip()
                self.data_list.append(
                    {
                        "img": os.path.join(self.data_root, "%s/images/%s" % (split, name)),
                        "label": os.path.join(self.data_root, "%s/labels/%s" % (split, name.split('.')[0] + '.png')),
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

        image = Image.open(datafiles["img"].replace('\\','/')).convert('RGB')

        label = cv2.imread(datafiles["label"].replace('\\','/'), 0).astype(np.int32)
        if self.tgt:
            label_mask = None
            if self.split == 'train':
                label_mask = np.array(Image.open(datafiles["label_mask"]), dtype=np.uint8)
            else:
                label_mask = np.ones_like(label, dtype=np.uint8) * 255

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

        # re-assign labels
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.uint8)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = np.array(label_copy, dtype=np.uint8)


        if self.split == 'test' or self.split == 'val':
            image, label = normalize(image, label)
            return  {
                "img": image,  # data
                'label': label,  # for test
                'name': datafiles['name'],  # for test to store the results
            }

        if self.tgt:
            origin_label = torch.from_numpy(deepcopy(label)).long()

            label = Image.fromarray(label)
            label_mask = Image.fromarray(label_mask)
            #if self.transform is not None:
            image, label, label_mask = resize(image, label, (0.5, 2.0), label_mask=label_mask)
            image, label, label_mask = crop(image, label, self.size[0], self.ignore_label, label_mask=label_mask)
            image, label, label_mask = hflip(image, label, p=0.5, label_mask=label_mask)

            h, w = label.size[1], label.size[0]

            img_w, img_s1, img_s2 = deepcopy(image), deepcopy(image), deepcopy(image)
            del image
            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)(img_s1)
            img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
            img_s1 = blur(img_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

            if random.random() < 0.8:
                img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)(img_s2)
            img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
            img_s2 = blur(img_s2, p=0.5)
            cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

            ignore_mask = Image.fromarray(np.zeros((label.size[1], label.size[0])))
            img_s1, ignore_mask = normalize(img_s1, ignore_mask)
            img_s2 = normalize(img_s2)

            label = torch.from_numpy(np.array(label)).long()
            label_mask = torch.from_numpy(np.array(label_mask)).long()
            ignore_mask[label == 255] = 255

            img_w = normalize(img_w)


        else:
            label = Image.fromarray(label)
            image, label = resize(image, label, (0.5, 2.0))
            image, label = crop(image, label, self.size[0], self.ignore_label)
            image, label = hflip(image, label, p=0.5)

            
            if self.src_aug:
                if random.random() < 0.8:
                    image = transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)(image)
                image = transforms.RandomGrayscale(p=0.2)(image)

            image, label = normalize(image, label)




        if self.tgt:
            ret_data = {
                "img": img_w,  # data
                'label': label,  # for test
                'mask': label_mask,  # for train
                'name': datafiles['name'],  # for test to store the results
                'path_to_mask': datafiles['label_mask'],  # for active to store new mask
                'path_to_indicator': datafiles['indicator'],  # store new indicator
                'size': torch.tensor([h, w]),  # for active to interpolate the output to original size
                'origin_mask': origin_mask,  # mask without transforms for active
                'origin_label': origin_label,  # label without transforms for active
                'active': active_indicator,
                'selected': active_selected,
                "img_s1": img_s1,
                "img_s2": img_s2,
                "ignore_mask": ignore_mask,
                "cutmix_box1": cutmix_box1,
                "cutmix_box2": cutmix_box2,
                "index": index,
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
