import os
import glob

import paddle

from transforms import Compose


class Dataset(paddle.io.Dataset):
    def __init__(self, dataset_root=None, transforms=None, mode = "train"):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = dataset_root
        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None
        self.mode = mode
        
        self.im1_list = [path.strip() for path in open(self.dataset_root, 'r').readlines()]
        self.im2_list = [path.replace("images", "gts") for path in self.im1_list]

        self.im1_list.sort()
        self.im2_list.sort()
        assert len(self.im1_list) == len(self.im2_list)

    def __getitem__(self, index):
        
        im1 = self.im1_list[index]
        im2 = self.im2_list[index]
        if self.transforms is not None:
            return self.transforms(im1, im2)
        else:
            return im1, im2

    def __len__(self):
        return len(self.im1_list)

if __name__ == '__main__':
    # dataset = Dataset(dataset_root=" ../data/data120844/moire_train_dataset")
    dataset = Dataset(dataset_root="../data/data120844/moire_train_dataset")
    print(len(dataset))
    for d in dataset:
        pass




