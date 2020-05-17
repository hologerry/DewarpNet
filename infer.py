# test end to end benchmark data test
import argparse
import os

import cv2
# import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch
# import torch.nn as nn
import torch.nn.functional as F

from models import get_model
# from utils import convert_state_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwarp(img, bm):
    w, h = img.shape[0], img.shape[1]
    bm = bm.transpose(1, 2).transpose(2, 3).detach().cpu().numpy()[0, :, :, :]
    bm0 = cv2.blur(bm[:, :, 0], (3, 3))
    bm1 = cv2.blur(bm[:, :, 1], (3, 3))
    bm0 = cv2.resize(bm0, (h, w))
    bm1 = cv2.resize(bm1, (h, w))
    bm = np.stack([bm0, bm1], axis=-1)
    bm = np.expand_dims(bm, 0)
    bm = torch.from_numpy(bm).double()

    img = img.astype(float) / 255.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).double()

    res = F.grid_sample(input=img, grid=bm)
    res = res[0].numpy().transpose((1, 2, 0))

    return res


def test(args, img_path, fname):
    bm_n_classes = 2

    bm_img_size = (128, 128)

    # Setup image
    print("Read Input Image from : {}".format(img_path))

    imgorg = cv2.imread(img_path)
    imgorg = cv2.cvtColor(imgorg, cv2.COLOR_BGR2RGB)
    img = cv2.resize(imgorg, bm_img_size)
    # img = img[:, :, ::-1]
    img = img.astype(float) / 255.0
    img = img.transpose(2, 0, 1)  # NHWC -> NCHW
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()

    img_bm_model = get_model('unet', bm_n_classes, in_channels=3)
    img_bm_model_state = torch.load(args.model_path, map_location=device)['model_state']
    img_bm_model.load_state_dict(img_bm_model_state)
    img_bm_model.eval()

    img_bm_model = img_bm_model.to(device)
    images = img.to(device)

    with torch.no_grad():
        outputs_bm = img_bm_model(images)

        # call unwarp
        uwpred = unwarp(imgorg, outputs_bm)

        if args.show:
            f1, axarr1 = plt.subplots(1, 2)
            axarr1[0].imshow(imgorg)
            axarr1[1].imshow(uwpred)
            plt.savefig(dpi=300)

        # Save the output
        outp = os.path.join(args.out_path, fname)
        cv2.imwrite(outp, uwpred[:, :, ::-1]*255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', type=str,
                        default='',
                        help='Path to the saved image bm model')
    parser.add_argument('--img_path', type=str, default='./eval/input/',
                        help='Path of the input image')
    parser.add_argument('--out_path', type=str, default='./eval/unwarp/',
                        help='Path of the output unwarped image')
    parser.add_argument('--show', dest='show', action='store_true',
                        help='Show the input image and output unwarped')
    parser.set_defaults(show=False)
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    for fname in os.listdir(args.img_path):
        if '.jpg' in fname or '.JPG' in fname or '.png' in fname:
            img_path = os.path.join(args.img_path, fname)
            test(args, img_path, fname)


# python infer.py --model_path ./eval/models/unetnc_doc3d.pkl --show
