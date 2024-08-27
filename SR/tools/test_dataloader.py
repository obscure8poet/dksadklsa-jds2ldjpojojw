import os
import glob
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms as T


class TestDataset:
    def __init__(self, args):
        """Initialize and preprocess the B100 dataset."""
        self.data_root = args.data_root
        self.image_scale = args.up_scale
        self.dataset_name = "test"
        self.subffix = args.img_subffix
        self.dataset = []
        self.pointer = 0
        self.batch_size = 16
        self.__preprocess__()
        self.num_images = len(self.dataset)

        c_transforms = []
        c_transforms.append(T.ToTensor())
        c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        self.img_transform = T.Compose(c_transforms)

    def __preprocess__(self):
        """Preprocess the B100 dataset."""

        # HR_files  = os.path.join(self.data_root, "mydata", "test", "hr")
        # LR_files  = os.path.join(self.data_root, "mydata", "test", "LR")

        # 测试ccpd
        # HR_files  = os.path.join(self.data_root,  "hr")
        # LR_files  = os.path.join(self.data_root,  "lr")
        HR_files  = os.path.join(self.data_root, "LBLP", "LBLP", "test","hr")
        LR_files  = os.path.join(self.data_root, "LBLP", "LBLP", "test","lr","lr")
        print("Evaluation data LR path: %s"%HR_files)
        print("Evaluation data LR path: %s"%LR_files)
        assert os.path.exists(HR_files)
        assert os.path.exists(LR_files)

        print("processing %s images..." % self.dataset_name)
        files = os.listdir(HR_files)
        for file in files:
            hr_path = os.path.join(HR_files, file)
            lr_path = os.path.join(LR_files, file)
            self.dataset.append([hr_path, lr_path])

        print('Finished preprocessing the %s dataset, total image number: %d...' % (self.dataset_name, len(self.dataset)))

    def __call__(self):
        """Return one batch images."""
        if self.pointer >= self.num_images:
            self.pointer = 0
            a = "The end of the story!"
            raise StopIteration(print(a))
        filename = self.dataset[self.pointer][0]
        image = Image.open(filename)
        hr = self.img_transform(image)
        filename = self.dataset[self.pointer][1]
        image = Image.open(filename)
        lr = self.img_transform(image)
        file_name = os.path.basename(filename)
        file_name = os.path.splitext(file_name)[0]
        hr_ls = hr.unsqueeze(0)
        lr_ls = lr.unsqueeze(0)
        nm_ls = [file_name, ]

        self.pointer += 1
        return hr_ls, lr_ls, nm_ls

    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'