# implementation of reconstruction loss for cropped images (no min max norm)
import torch
import torch.nn as nn
import torch.nn.functional as F
import ssim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unwarp(img, bm):

    # print(bm.type)
    # img = torch.from_numpy(img).cuda().double()
    n, c, h, w = img.shape
    # resize bm to img size
    # print bm.shape
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True)
    # print bm.shape

    # img = img.double()
    bm = bm.transpose(1, 2).transpose(2, 3)  # NCHW -> NHWC
    res = F.grid_sample(input=img, grid=bm)
    return res


class UnwarpLoss(torch.nn.Module):
    def __init__(self):
        super(UnwarpLoss, self).__init__()
        # self.xmx, self.xmn, self.ymx, self.ymn = 166.28639310649825, -3.792634897181367, 189.04606710275974, -18.982843029373125
        # self.xmx, self.xmn, self.ymx, self.ymn = 434.8578833991327, 14.898654260467202, 435.0363953546216, 14.515746051497239
        # self.xmx, self.xmn, self.ymx, self.ymn = 434.9877152088082, 14.546402972133514, 435.0591952709043, 14.489902537540008
        # self.xmx, self.xmn, self.ymx, self.ymn = 435.14545757153445, 13.410177297916455, 435.3297804574046, 14.194541402379988
        # self.xmx, self.xmn, self.ymx, self.ymn = 0.0, 0.0, 0.0, 0.0
        self.l2_loss_fn = nn.MSELoss().to(device)
        self.ssim_loss_fn = ssim.SSIM().to(device)

    def forward(self, inp_img, pred_bm, ground_bm):
        # image [n,c,h,w], target_nhwc [n,h,w,c], labels [n,h,w,c]
        # n, c, h, w = inp.shape  # this has 6 channels if image is passed
        # print (h,w)
        # inp=inp.detach().cpu().numpy()
        # inp_img = inp_img[:, :-1, :, :]  # img in bgr

        # denormalize pred_bm
        # pred_bm=(pred_bm/2.0)+0.5
        # pred_bm[:,:,:,0]=(pred_bm[:,:,:,0]*(self.xmx-self.xmn)) +self.xmn
        # pred_bm[:,:,:,1]=(pred_bm[:,:,:,1]*(self.ymx-self.ymn)) +self.ymn
        # pred_bm[:,:,:,0]=pred_bm[:,:,:,0]/float(448.0)
        # pred_bm[:,:,:,1]=pred_bm[:,:,:,1]/float(448.0)
        # pred_bm=(pred_bm-0.5)*2
        # no need denormalize, in favor of grid_sample
        # pred_bm = pred_bm.double()

        # denormalize ground_bm
        # ground_bm=(ground_bm/2.0)+0.5
        # ground_bm[:,:,:,0]=(ground_bm[:,:,:,0]*(self.xmx-self.xmn)) +self.xmn
        # ground_bm[:,:,:,1]=(ground_bm[:,:,:,1]*(self.ymx-self.ymn)) +self.ymn
        # ground_bm[:,:,:,0]=ground_bm[:,:,:,0]/float(448.0)
        # ground_bm[:,:,:,1]=ground_bm[:,:,:,1]/float(448.0)
        # ground_bm=(ground_bm-0.5)*2

        # no need denormalize, in favor of grid_sample
        # ground_bm = ground_bm.double()

        uwpred = unwarp(inp_img, pred_bm)
        uwground = unwarp(inp_img, ground_bm)

        l2_loss = self.l2_loss_fn(uwpred, uwground)
        ssim_loss = 1 - self.ssim_loss_fn(uwpred, uwground)

        # print(uloss)
        # del pred_bm
        # del ground_bm
        # del inp_img

        return l2_loss, ssim_loss, uwpred, uwground
