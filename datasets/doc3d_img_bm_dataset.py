import collections
from os.path import join as pjoin

import cv2
import hdf5storage as h5
import numpy as np
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms

# from datasets.augmentationsk import data_aug, tight_crop


class IMGBMDataset(data.Dataset):
    """
    Dataset for RGB Image -> Backward Mapping.
    """

    def __init__(self, root, split='train', is_transform=False,
                 img_size=512, augmentations=None):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 2  # target number of channel
        self.files = collections.defaultdict(list)
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)

        for split in ['train', 'val']:
            path = pjoin(self.root, split + '.txt')
            file_list = tuple(open(path, 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list
        # self.setup_annotations()
        # if self.augmentations:
        #     self.txpths = []
        #     with open(os.path.join(self.root[:-7], 'augtexnames.txt'), 'r') as f:
        #         for line in f:
        #             txpth = line.strip()
        #             self.txpths.append(txpth)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        # 1/824_8-cp_Page_0503-7Nw0001
        im_name = self.files[self.split][index]

        im_path = pjoin(self.root, 'img', im_name + '.png')
        im = Image.open(im_path).convert('RGB')

        bm_path = pjoin(self.root, 'bm', im_name + '.mat')
        bm = h5.loadmat(bm_path)['bm']

        # chess = 'chess48'
        # checker_path = pjoin(self.root, 'recon', im_name[:-4]+chess+'0001.png')
        # chk = misc.imread(checker_path, mode='RGB')

        # if 'val' in self.split:
        #     im, lbl = tight_crop(im/255.0, lbl)
        # if self.augmentations:  # this is for training, default false for validation\
        #     tex_id = random.randint(0, len(self.txpths)-1)
        #     txpth = self.txpths[tex_id]
        #     tex = cv2.imread(os.path.join(self.root[:-7], txpth)).astype(np.uint8)
        #     bg = cv2.resize(tex, self.img_size, interpolation=cv2.INTER_NEAREST)
        #     im, lbl = data_aug(im, lbl, bg)
        if self.is_transform:
            image, label = self.transform(im, bm)
        return image, label

    def transform(self, img, bm, chk=None):
        # img = misc.imresize(img, self.img_size)  # uint8 with RGB mode
        img = img.resize(self.img_size)
        # if img.shape[-1] == 4:
        #     img = img[:, :, :3]   # Discard the alpha channel
        # # img = img[:, :, ::-1]  # RGB -> BGR
        # img = img.astype(float) / 255.0
        # img = img.transpose(2, 0, 1)  # NHWC -> NCHW

        # currently drop checkerboard
        # chk = misc.imresize(chk, self.img_size)
        # chk = chk[:, :, ::-1]  # RGB -> BGR
        # chk = chk.astype(np.float64)
        # if chk.shape[2] == 4:
        #     chk = chk[:, :, :3]
        # chk = chk.astype(float) / 255.0
        # chk = chk.transpose(2, 0, 1)  # NHWC -> NCHW

        bm = bm.astype(float)
        # normalize label [-1,1]
        bm = bm / np.array([448.0, 448.0])
        bm = (bm - 0.5) * 2
        bm0 = cv2.resize(bm[:, :, 0], (self.img_size[0], self.img_size[1]))
        bm1 = cv2.resize(bm[:, :, 1], (self.img_size[0], self.img_size[1]))

        label = np.stack([bm0, bm1], axis=0)

        # to torch
        # image = np.concatenate([img, chk], axis=0)
        # image = torch.from_numpy(img).float()
        image = transforms.ToTensor()(img)
        label = torch.from_numpy(label).float()  # NCHW

        return image, label
