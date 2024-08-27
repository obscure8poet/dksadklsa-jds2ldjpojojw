
import os
import cv2
import glob
import torch
from tqdm import tqdm

class EvalDataset:
    def __init__(self, args):

        """Initialize and preprocess the urban100 data."""
        self.data_root      = args.data_root
        self.image_scale    = args.up_scale
        self.dataset_name   = "test"
        self.subffix        = args.img_subffix
        self.dataset        = []
        self.pointer        = 0
        self.batch_size     = 1

        self.dataset_name == "eval"

        self.__preprocess__()
        self.num_images     = len(self.dataset)

    def __preprocess__(self):

        HR_files  = os.path.join(self.data_root, "mydata", "test", "hr")
        LR_files  = os.path.join(self.data_root, "mydata", "test", "LR")

        print("Evaluation data LR path: %s"%HR_files)
        print("Evaluation data LR path: %s"%LR_files)
        assert os.path.exists(HR_files)
        assert os.path.exists(LR_files)

        files = os.listdir(HR_files)
        data_paths = []
        for file in files:
            hr_path = os.path.join(HR_files, file)
            lr_path = os.path.join(LR_files, file)
            data_paths.append([hr_path, lr_path])
        
        for item_pair in tqdm(data_paths):
            # print(item_pair[0])
            hr_img      = cv2.imread(item_pair[0])
            hr_img      = cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB)
            hr_img      = hr_img.transpose((2,0,1))
            hr_img      = torch.from_numpy(hr_img)
            
            lr_img      = cv2.imread(item_pair[1])
            lr_img      = cv2.cvtColor(lr_img,cv2.COLOR_BGR2RGB)
            lr_img      = lr_img.transpose((2,0,1))
            lr_img      = torch.from_numpy(lr_img)
            
            self.dataset.append((hr_img,lr_img))

        print('Finished preprocessing the Urban100 Validation data, total image number: %d...'%len(self.dataset))

    def __call__(self):
        """Return one batch images."""
        if self.pointer>=self.num_images:
            self.pointer = 0
        hr = self.dataset[self.pointer][0]
        lr = self.dataset[self.pointer][1]
        hr = (hr/255.0 - 0.5) * 2.0
        lr = (lr/255.0 - 0.5) * 2.0
        hr = hr.unsqueeze(0)
        lr = lr.unsqueeze(0)
        self.pointer += 1
        return hr, lr
    
    def __len__(self):
        return self.num_images

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.data_root + ')'