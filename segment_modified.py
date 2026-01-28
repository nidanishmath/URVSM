import os
from os.path import exists
import argparse

import torch
import cv2 as cv
import numpy as np

from models.segmentation import UNet_vanilla, ResDO_UNet

class DriveDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img = cv.imread(os.path.join(self.img_dir, name))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img, name


parser = argparse.ArgumentParser()

# =========================
# Essential
# =========================
parser.add_argument('--datapath', default='./data/images', help='path to input images')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--note', type=str, default='experiment_name')

# =========================
# Segmentation model
# =========================
parser.add_argument('--seg_model', default='Resdounet',
                    choices=['unet_vanilla', 'Resdounet'])

parser.add_argument('--segnet_checkpoint_unet',
                    default='./ckpt/segmentation.pth',
                    help='path to UNet checkpoint')

parser.add_argument('--segnet_checkpoint_resdounet',
                    default='./ckpt/segmentation_resdo.pth',
                    help='path to ResDO-UNet checkpoint')

# =========================
# Save paths
# =========================
parser.add_argument('--save_segmentation', default='./result',
                    help='root path to save segmentation results')


def main():
    args = parser.parse_args()

    # Device
    args.device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )

    # Output directory
    args.save_segmentation = os.path.join(
        args.save_segmentation, args.note, 'segmentation'
    )
    if not exists(args.save_segmentation):
        os.makedirs(args.save_segmentation)

    # =========================
    # Data loader
    # =========================
    eval_dataset = Retinal_loader(args.datapath)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Dataloader ready. Images:", len(eval_dataset))

    # =========================
    # Load segmentation model
    # =========================
    if args.seg_model == 'unet_vanilla':
        seg_net = UNet_vanilla()
        seg_net.load_state_dict(
            torch.load(args.segnet_checkpoint_unet, map_location=args.device)
        )

    elif args.seg_model == 'Resdounet':
        seg_net = ResDO_UNet(in_ch=4, out_ch=1)
        seg_net.load_state_dict(
            torch.load(args.segnet_checkpoint_resdounet, map_location=args.device)
        )
    else:
        raise ValueError('Unknown segmentation model')

    seg_net = seg_net.to(args.device)
    seg_net.eval()

    # =========================
    # Inference
    # =========================
    with torch.no_grad():
        for i, (sample, name) in enumerate(eval_loader):
            print("Processing:", name[0])

            # Ensure RGB
            if sample.dim() == 3:
                sample = sample.unsqueeze(-1).repeat(1, 1, 1, 3)

            sample = sample.permute(0, 3, 1, 2).float().to(args.device)
            N, C, H, W = sample.shape
            assert N == 1

            # Initial empty segmentation
            init_seg = torch.zeros((N, 1, H, W), device=args.device)

            # URVSM iterative refinement (NO translation)
            net_in = torch.cat((init_seg, sample), dim=1)
            y = seg_net(net_in)

            for _ in range(1):
                net_in = torch.cat((y, sample), dim=1)
                y = seg_net(net_in)

            # Save output
            out = y[0, 0].cpu().numpy()
            out = (out * 255).astype(np.uint8)

            cv.imwrite(
                os.path.join(args.save_segmentation, f"{name[0]}"),
                out
            )

    print("Segmentation completed successfully!")


if __name__ == '__main__':
    main()
