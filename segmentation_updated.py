import os
from os.path import exists
import argparse
import glob

import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np

from models.segmentation import UNet_vanilla, ResDO_UNet

parser = argparse.ArgumentParser()

# Essential
parser.add_argument('--drive_path', default='./DRIVE', help='path to DRIVE dataset')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--note', type=str, default='drive_finetune')

# mode
parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'])

# training params
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for fine-tuning')
parser.add_argument('--epochs', type=int, default=100, help='number of fine-tuning epochs')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')

# model
parser.add_argument('--seg_model', default='Resdounet', type=str, choices=['unet_vanilla', 'Resdounet'])
parser.add_argument('--segnet_checkpoint_unet', default='./ckpt/segmentation.pth', help='path to segmentation model checkpoint for unet_vanilla')
parser.add_argument('--segnet_checkpoint_resdounet', default='./ckpt/segmentation_resdo.pth', help='path to segmentation model checkpoint for Resdounet')
parser.add_argument('--save_ckpt', default='./ckpt/finetuned_resdo.pth', help='path to save fine-tuned checkpoint')

# save
parser.add_argument('--save_segmentation', default='./result', help='path to save segmentation results')
parser.add_argument('--save_vis', default='./result', help='path to save visualizations')

class DRIVE_loader(torch.utils.data.Dataset):
    def __init__(self, root, split='training', is_transform=False):
        self.root = root
        self.split = 'training' if split == 'training' else 'test'
        self.is_transform = is_transform
        folder = self.split
        image_pattern = f'*{self.split}.tif'
        mask_pattern = f'*{self.split}_mask.gif'
        manual_folder = '1st_manual'
        self.image_paths = sorted(glob.glob(os.path.join(root, folder, 'images', image_pattern)))
        self.manual_paths = sorted(glob.glob(os.path.join(root, folder, manual_folder, '*_manual1.gif')))
        self.mask_paths = sorted(glob.glob(os.path.join(root, folder, 'mask', mask_pattern)))
        assert len(self.image_paths) == len(self.manual_paths) == len(self.mask_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # H W 3 uint8
        gt_path = self.manual_paths[idx]
        gt = cv.imread(gt_path, cv.IMREAD_GRAYSCALE) / 255.0  # H W float 0-1
        name = os.path.basename(img_path).split('_')[0]
        if self.is_transform:
            # Add data augmentations here if needed (e.g., flips, rotations using torch transforms or cv2)
            pass
        image = torch.from_numpy(image)  # H W 3 torch.float() will be done later
        gt = torch.from_numpy(gt).unsqueeze(0)  # 1 H W
        return image, gt.float(), name

def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3,3), 1, 1)
    return p1

def soft_dilate(img):
    return F.max_pool2d(img, (3,3), 1, 1)

def soft_open(img):
    return soft_dilate(soft_erode(img))

def soft_skel(img, iter_):
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for j in range(iter_):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel

def dice_loss(pred, target):
    smooth = 1.
    pred = torch.sigmoid(pred)
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def cl_dice_loss(pred, target, iter_=20):
    pred = torch.sigmoid(pred)
    target = target.float()
    s_pred = soft_skel(pred, iter_)
    s_target = soft_skel(target, iter_)
    tprec = ( (s_pred * target).sum(dim=[1,2,3]) / (s_pred.sum(dim=[1,2,3]) + 1e-10) ).mean()
    tsens = ( (s_target * pred).sum(dim=[1,2,3]) / (s_target.sum(dim=[1,2,3]) + 1e-10) ).mean()
    cl_dice = 2. * tprec * tsens / (tprec + tsens + 1e-10)
    return 1 - cl_dice

def main():
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu_id)) if torch.cuda.is_available() else torch.device('cpu')

    args.save_segmentation = os.path.join(args.save_segmentation, args.note, 'segmentation')
    if not exists(args.save_segmentation):
        os.makedirs(args.save_segmentation)
    args.save_vis = os.path.join(args.save_vis, args.note, 'vis')
    if not exists(args.save_vis):
        os.makedirs(args.save_vis)

    # Load segmentation model
    if args.seg_model == 'unet_vanilla':
        seg_net = UNet_vanilla()
        seg_net.load_state_dict(torch.load(args.segnet_checkpoint_unet))
    elif args.seg_model == 'Resdounet':
        seg_net = ResDO_UNet(in_ch=4, out_ch=1)
        seg_net.load_state_dict(torch.load(args.segnet_checkpoint_resdounet))
    else:
        raise ValueError('Specified segmentation backbone unavailable.')
    seg_net = seg_net.to(args.device)

    if args.mode == 'train':
        train_dataset = DRIVE_loader(args.drive_path, split='training', is_transform=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_dataset = DRIVE_loader(args.drive_path, split='test', is_transform=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        optimizer = torch.optim.Adam(seg_net.parameters(), lr=args.lr)
        bce_criterion = torch.nn.BCEWithLogitsLoss()

        for epoch in range(args.epochs):
            seg_net.train()
            train_loss = 0.0
            num_batches = 0
            for sample, gt, name in train_loader:
                sample = sample.permute(0, 3, 1, 2).to(torch.float32).to(args.device) / 255.0  # N C H W 0-1
                gt = gt.to(args.device)  # N 1 H W 0-1
                optimizer.zero_grad()
                init_seg = torch.zeros_like(gt)
                net_in = torch.cat((init_seg, sample), dim=1)  # N 4 H W
                y = seg_net(net_in)
                for i in range(1):
                    net_in = torch.cat((y, sample), dim=1)
                    y = seg_net(net_in)
                bce = bce_criterion(y, gt)
                dice = dice_loss(y, gt)
                if epoch < 25:
                    loss = bce + dice  # BCE + DiceRecover (assuming DiceRecover is standard Dice)
                else:
                    cldice = cl_dice_loss(y, gt)
                    loss = bce + dice + cldice  # BCE + Dice + clDiceEnforce
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss / num_batches:.4f}")

            # Validation
            seg_net.eval()
            val_loss = 0.0
            num_batches = 0
            with torch.no_grad():
                for sample, gt, name in val_loader:
                    sample = sample.permute(0, 3, 1, 2).to(torch.float32).to(args.device) / 255.0
                    gt = gt.to(args.device)
                    N = sample.size(0)
                    init_seg = torch.zeros([N, 1, *sample.shape[2:]], dtype=torch.float32, device=args.device)
                    net_in = torch.cat((init_seg, sample), dim=1)
                    y = seg_net(net_in)
                    for i in range(1):
                        net_in = torch.cat((y, sample), dim=1)
                        y = seg_net(net_in)
                    bce = bce_criterion(y, gt)
                    dice = dice_loss(y, gt)
                    if epoch < 25:
                        loss = bce + dice
                    else:
                        cldice = cl_dice_loss(y, gt)
                        loss = bce + dice + cldice
                    val_loss += loss.item()
                    num_batches += 1
            print(f"Epoch {epoch + 1}/{args.epochs}, Val Loss: {val_loss / num_batches:.4f}")

        torch.save(seg_net.state_dict(), args.save_ckpt)
        print(f"Fine-tuned model saved to {args.save_ckpt}")

    elif args.mode == 'eval':
        eval_dataset = DRIVE_loader(args.drive_path, split='test', is_transform=False)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)

        seg_net.eval()
        with torch.no_grad():
            for sample, gt, name in eval_loader:
                print(name[0])
                sample = sample.permute(0, 3, 1, 2).to(torch.float32).to(args.device) / 255.0  # 1 C H W
                N, C, H, W = sample.size()
                init_seg = torch.zeros([N, 1, H, W], dtype=torch.float32, device=args.device)
                net_in = torch.cat((init_seg, sample), dim=1)
                y = seg_net(net_in)
                # First stage visualization
                out_first = torch.sigmoid(y).squeeze(0).squeeze(0).detach().cpu().numpy()
                cv.imwrite(os.path.join(args.save_vis, f'{name[0]}_pred_first.png'), (out_first * 255).astype(np.uint8))
                # Refinement
                for i in range(1):
                    net_in = torch.cat((y, sample), dim=1)
                    y = seg_net(net_in)
                # Final output
                out = torch.sigmoid(y).squeeze(0).squeeze(0).detach().cpu().numpy()
                cv.imwrite(os.path.join(args.save_segmentation, f'{name[0]}.png'), (out * 255).astype(np.uint8))
                cv.imwrite(os.path.join(args.save_vis, f'{name[0]}_pred_final.png'), (out * 255).astype(np.uint8))
                # Save original image
                orig = sample.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255
                cv.imwrite(os.path.join(args.save_vis, f'{name[0]}_orig.png'), orig.astype(np.uint8))
                # Save ground truth
                gt_save = gt.squeeze(0).squeeze(0).detach().cpu().numpy() * 255
                cv.imwrite(os.path.join(args.save_vis, f'{name[0]}_gt.png'), gt_save.astype(np.uint8))


if __name__ == '__main__':
    main()