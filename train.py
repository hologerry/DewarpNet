# code to train backward mapping regression from GT world coordinates
# models are saved in checkpoints-bm/
import argparse
import os

import torch
# import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils import data
from tqdm import tqdm

from unwarp_loss import UnwarpLoss
from docunet_loss import DocUnetLoss
from smoothness_loss import SmoothnessLoss
from datasets import get_dataset
from models import get_model
from utils import get_lr, show_unwarp_tnsboard


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def write_log_file(log_file, losses, epoch, lr, phase):
    # log_file.write(f"\n{phase} LRate: {lr} Epoch: {epoch} Loss: {losses[0]} MSE: {losses[1]} UnwarpL2: {losses[2]} UnwarpSSIM: {losses[3]}\n")
    # log_file.write(f"\n{phase} LRate: {lr} Epoch: {epoch} DocUnetLoss: {losses[0]} ReconLoss: {losses[1]}\n")
    log_file.write(f"\n{phase} LRate: {lr} Epoch: {epoch} DocUnetLoss: {losses[0]} ReconLoss: {losses[1]} SmoothLoss: {losses[2]}\n")


def train(args):
    dataset = get_dataset('doc3dimgbm')
    data_path = args.data_path
    train_dataset = dataset(data_path, is_transform=True,
                            img_size=(args.img_rows, args.img_cols))
    val_dataset = dataset(data_path, is_transform=True,
                          split='val', img_size=(args.img_rows, args.img_cols))

    n_classes = train_dataset.n_classes
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size)

    # Setup Model
    model = get_model(args.arch, n_classes, in_channels=3)
    model = torch.nn.DataParallel(model)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=True)

    # LR Scheduler
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Losses
    # l1_criterion = nn.L1Loss().to(device)
    # l2_criterion = nn.MSELoss().to(device)
    unwarp_criterion = UnwarpLoss().to(device)
    docunet_criterion = DocUnetLoss().to(device)
    smooth_criterion = SmoothnessLoss().to(device)

    epoch_start = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"Loading model and optimizer from checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
            epoch_start = checkpoint['epoch']
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    # Log file:
    experiment_dir = os.path.join('experiments', args.experiment_name)
    checkpoint_dir = os.path.join('experiments', args.experiment_name, 'checkpoint')
    tboard_dir = os.path.join('experiments', args.experiment_name, 'log')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(tboard_dir):
        os.makedirs(tboard_dir)
    # network_activation(t=[-1,1])_dataset_lossparams_augmentations_trainstart

    log_file_name = os.path.join(experiment_dir, 'loss_log.txt')
    if os.path.isfile(log_file_name):
        log_file = open(log_file_name, 'a')
    else:
        log_file = open(log_file_name, 'w+')

    log_file.write('\n---------------  '+args.experiment_name+'  ---------------\n')
    log_file.write('options:\n')
    log_file.write(str(args))

    # Setup tensorboard for visualization
    if args.tboard:
        # save tensorboard logs in runs/<experiment_name>
        writer = SummaryWriter(tboard_dir)

    # best_val_uwarpssim = 99999.0
    best_val_mse_loss = 99999.0
    global_step = 0

    for epoch in range(epoch_start, args.n_epoch):
        avg_loss = 0.0
        # avg_l1_loss = 0.0
        avg_recon_loss = 0.0
        # avg_ssim_loss = 0.0
        # avg_mse_loss = 0.0
        avg_docunet_loss = 0.0
        avg_smooth_loss = 0.0
        model.train()
        for i, (images, gt_bms) in enumerate(train_loader):
            images = images.to(device)
            gt_bms = gt_bms.to(device)
            optimizer.zero_grad()
            pred_bms = model(images)
            # l1_loss = l1_criterion(pred_bms, gt_bms)
            # l2_loss = l2_criterion(pred_bms, gt_bms)
            docunet_loss = docunet_criterion(pred_bms, gt_bms)
            smooth_loss = smooth_criterion(pred_bms)
            recon_loss, ssim_loss, uwpredict, uwground = unwarp_criterion(images, pred_bms, gt_bms)
            # loss = (10.0 * l1_loss) + (5.0 * l2_loss) + (5.0 * recon_loss)  # + (3.0 * ssim_loss)
            loss = docunet_loss + 0.01 * recon_loss + 0.01 * smooth_loss
            # loss = l1_loss
            avg_loss += loss.item()
            # avg_l1_loss += l1_loss.item()
            avg_recon_loss += recon_loss.item()
            # avg_ssim_loss += ssim_loss.item()
            # avg_mse_loss += l2_loss.item()
            avg_docunet_loss += docunet_loss.item()
            avg_smooth_loss += smooth_loss.item()

            loss.backward()
            optimizer.step()
            global_step += 1

            if (i+1) % 20 == 0:
                avg_loss = avg_loss / 20
                msg = f"Epoch[{epoch+1}/{args.n_epoch}] Batch [{i+1}/{len(train_loader)}] Loss: {avg_loss:.4f}"
                print(msg)
                log_file.write(msg+'\n')
                avg_loss = 0.0

            if args.tboard and (i+1) % 20 == 0:
                show_unwarp_tnsboard(global_step, writer, images.detach().cpu(), uwpredict.detach().cpu(), uwground.detach().cpu(), 8,
                                     'Train Input', 'Train Pred Unwarp', 'Train GT Unwarp')
                writer.add_scalar('AVG Loss/train', avg_loss/(i+1), global_step)
                # writer.add_scalar('L1 Loss/train', avg_l1_loss/(i+1), global_step)
                writer.add_scalar('Recon Loss/train', avg_recon_loss/(i+1), global_step)
                writer.add_scalar('Docunet Loss/train', avg_docunet_loss/(i+1), global_step)
                writer.add_scalar('Smooth Loss/train', avg_smooth_loss/(i+1), global_step)
                # writer.add_scalar('SSIM Loss/train', avg_ssim_loss/(i+1), global_step)

        # avg_ssim_loss = avg_ssim_loss/len(train_loader)
        # avg_loss = avg_loss/len(train_loader)
        avg_recon_loss = avg_recon_loss/len(train_loader)
        # avg_l1_loss = avg_l1_loss/len(train_loader)
        # avg_mse_loss = avg_mse_loss/len(train_loader)
        avg_docunet_loss = avg_docunet_loss/len(train_loader)
        avg_smooth_loss = avg_smooth_loss/len(train_loader)
        print(f"Training avg loss: {avg_loss:.4f}")
        # print(f"Training L1: {avg_l1_loss:.4f}")
        # print(f"Training MSE: {avg_mse_loss}")

        train_losses = [avg_docunet_loss, avg_recon_loss, avg_smooth_loss]
        lr = get_lr(optimizer)
        write_log_file(log_file, train_losses, epoch+1, lr, 'Train')

        model.eval()
        # val_loss = 0.0
        # val_l1_loss = 0.0
        # val_mse_loss = 0.0
        val_recon_loss = 0.0
        # val_ssim_loss = 0.0
        val_docunet_loss = 0.0
        val_smooth_loss = 0.0

        for i_val, (images_val, gt_bms_val) in tqdm(enumerate(val_loader)):
            with torch.no_grad():
                images_val = images_val.to(device)
                gt_bms_val = gt_bms_val.to(device)
                pred_bms_val = model(images_val)
                # l1_loss_val = l1_criterion(pred_bms_val, gt_bms_val)
                # l2_loss_val = l2_criterion(pred_bms_val, gt_bms_val)
                recon_loss_val, ssim_loss_val, uwpred_val, uwground_val = unwarp_criterion(images_val, pred_bms_val, gt_bms_val)
                docunet_loss_val = docunet_criterion(pred_bms_val, gt_bms_val)
                smooth_loss_val = smooth_criterion(pred_bms_val)

                # val_l1_loss += l1_loss_val.item()
                val_recon_loss += recon_loss_val.item()
                # val_ssim_loss += ssim_loss_val.item()
                # val_mse_loss += l2_loss_val.item()
                val_docunet_loss += docunet_loss_val.item()
                val_smooth_loss += smooth_loss_val.item()
            if args.tboard:
                show_unwarp_tnsboard(epoch+1, writer, images_val.detach().cpu(), uwpred_val.detach().cpu(), uwground_val.detach().cpu(), 8,
                                     'Val Input', 'Val Pred Unwarp', 'Val GT Unwarp')

        # val_l1_loss = val_l1_loss/len(val_loader)
        # val_mse_loss = val_mse_loss/len(val_loader)
        # val_ssimloss = val_ssim_loss/len(val_loader)
        val_recon_loss = val_recon_loss/len(val_loader)
        val_docunet_loss = val_docunet_loss/len(val_loader)
        val_smooth_loss = val_smooth_loss/len(val_loader)
        # print(f"val loss at epoch {epoch+1}: {val_l1_loss}")
        # print(f"val mse: {val_mse_loss}")
        print(f"val docunet: {val_docunet_loss}")
        # val_losses = [val_l1_loss, val_mse_loss, val_recon_loss, val_ssim_loss]
        val_losses = [val_docunet_loss, val_recon_loss, val_smooth_loss]
        write_log_file(log_file, val_losses, epoch+1, lr, 'Val')
        if args.tboard:
            # log the val losses
            # writer.add_scalar('BM: L1 Loss/val', val_l1_loss, epoch+1)
            writer.add_scalar('Recon Loss/val', val_recon_loss, epoch+1)
            # writer.add_scalar('SSIM Loss/val', val_ssimloss, epoch+1)
            writer.add_scalar('DocUnet Loss/val', val_docunet_loss, epoch+1)
            writer.add_scalar('Smooth Loss/val', val_smooth_loss, epoch+1)

        # reduce learning rate
        sched.step(val_docunet_loss)

        if val_docunet_loss < best_val_mse_loss:
            best_val_mse_loss = val_docunet_loss
            state = {'epoch': epoch+1,
                     'model_state': model.module.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state, os.path.join(checkpoint_dir, f"{args.arch}_{epoch+1}_{val_docunet_loss}_best_model.pth"))

        if (epoch+1) % 10 == 0:
            state = {'epoch': epoch+1,
                     'model_state': model.module.state_dict(),
                     'optimizer_state': optimizer.state_dict(), }
            torch.save(state, os.path.join(checkpoint_dir, f"{args.arch}_{epoch+1}_{val_docunet_loss}_model.pth"))

    log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', type=str, default='unet',
                        help='Architecture to use [\'densenet, unet\']')
    parser.add_argument('--data_path', type=str, default='/root/doc3d',
                        help='Data path to load data')
    parser.add_argument('--img_rows', type=int, default=128,
                        help='Height of the input image')
    parser.add_argument('--img_cols', type=int, default=128,
                        help='Width of the input image')
    parser.add_argument('--experiment_name', type=str, default='unet_img_bm_smooth_loss')
    parser.add_argument('--n_epoch', type=int, default=100,
                        help='# of the epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning Rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to previous saved model to restart from')
    parser.add_argument('--tboard', type=bool, default=True,
                        help='Enable visualization(s) on tensorboard | False by default')

    args = parser.parse_args()
    train(args)

# CUDA_VISIBLE_DEVICES=1 python train.py --arch unetnc --dataset doc3d --img_rows 128 --img_cols 128 --img_norm --n_epoch 250
# --batch_size 50 --l_rate 0.0001 --tboard --data_path /root/doc3d
