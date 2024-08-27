import argparse
import os
import sys
sys.path.append("/")
import os
import cv2
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils import data


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.__preload__()

    def __preload__(self):
        try:
            self.hr, self.lr = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.hr, self.lr = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.hr = self.hr.cuda(non_blocking=True)
            self.lr = self.lr.cuda(non_blocking=True)
            self.hr = (self.hr / 255.0 - 0.5) * 2.0
            self.lr = (self.lr / 255.0 - 0.5) * 2.0

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        hr = self.hr
        lr = self.lr
        self.__preload__()
        return hr, lr

    def __len__(self):
        """Return the number of images."""
        return len(self.loader)


class DataLoader(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""
    def __init__(self, args):
        """Initialize and preprocess the flickr and train dataset."""
        self.root = args.data_root
        self.i_s = args.up_scale
        self.l_ps = args.patch_size
        self.h_ps = args.patch_size * args.up_scale
        self.d_e = args.dataset_enlarge
        self.subffix = args.img_subffix
        self.dataset = []
        self.random_seed = args.random_seed
        random.seed(self.random_seed)
        self.__preprocess__()
        self.num_images = len(self.pathes)

    def __preprocess__(self):
        """Preprocess the Artworks dataset."""
        data_path_list = []
        # 判断是训练集或者是测试集

        HR_files = os.path.join(self.root, "mydata", "train", "hr")
        LR_files = os.path.join(self.root, "mydata", "train", "LR")
        print("Traintion data LR path: %s"%HR_files)
        print("Traintion data LR path: %s"%LR_files)
        files = os.listdir(HR_files)

        for file in files:
            hr_path = os.path.join(HR_files, file)
            lr_path = os.path.join(LR_files, file)
            data_path_list.append([hr_path, lr_path])

        random.shuffle(data_path_list)
        print('Finished preprocessing the data dataset, total image number: %d...' % len(data_path_list))

        for item_pair in tqdm(data_path_list[:]):
            hr_img = cv2.imread(item_pair[0])
            hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
            hr_img = hr_img.transpose((2, 0, 1))
            hr_img = torch.from_numpy(hr_img)

            lr_img = cv2.imread(item_pair[1])
            # cv2.imshow("test", lr_img)
            # cv2.waitKey(0)  # 等待按键
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            lr_img = lr_img.transpose((2, 0, 1))
            lr_img = torch.from_numpy(lr_img)

            self.dataset.append((hr_img, lr_img))

        indeices = np.random.randint(0, len(self.dataset), size=self.d_e * len(self.dataset))
        self.pathes = indeices.tolist()
        print("Finish to read the dataset!")

    def __getitem__(self, index):

        hr_img = self.dataset[self.pathes[index]][0]
        lr_img = self.dataset[self.pathes[index]][1]

        hight = lr_img.shape[1]
        width = lr_img.shape[2]

        r_h = random.randint(0, hight - self.l_ps)
        r_w = random.randint(0, width - self.l_ps)


        hr_img = hr_img[:, r_h * self.i_s:(r_h * self.i_s + self.h_ps),
                 r_w * self.i_s:(r_w * self.i_s + self.h_ps)]
        lr_img = lr_img[:, r_h:(r_h + self.l_ps), r_w:(r_w + self.l_ps)]

        flip_ran = random.randint(0, 2)

        if flip_ran == 0:
            # horizontal
            hr_img = torch.flip(hr_img, [1])
            lr_img = torch.flip(lr_img, [1])

        elif flip_ran == 1:
            # vertical
            hr_img = torch.flip(hr_img, [2])
            lr_img = torch.flip(lr_img, [2])

        rot_ran = random.randint(0, 3)

        if rot_ran != 0:
            # horizontal
            hr_img = torch.rot90(hr_img, rot_ran, [1, 2])
            lr_img = torch.rot90(lr_img, rot_ran, [1, 2])
        # else:
        #     # 查看返回的图片
        #     hr_img = hr_img.numpy()
        #     hr_img = np.transpose(hr_img, (LR, 2, 0))
        #     cv2.imshow("hr_img", hr_img)
        #     cv2.waitKey(0)
        #
        #     lr_img = lr_img.numpy()
        #     lr_img = np.transpose(lr_img, (LR, 2, 0))
        #     cv2.imshow("hr_img", lr_img)
        #     cv2.waitKey(0)
        return hr_img, lr_img

    def __len__(self):
        return self.num_images

def GetLoader(args):

    content_dataset = DataLoader(args)

    content_data_loader = data.DataLoader(
        dataset=content_dataset,
        batch_size=args.batch_size,
        num_workers=args.dataloader_workers,
        drop_last=True,
        shuffle=True,
        pin_memory=True)

    prefetcher = DataPrefetcher(content_data_loader)

    return prefetcher