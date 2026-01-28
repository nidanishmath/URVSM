import os
from os.path import exists
import argparse

import torch
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader, Dataset

from models.segmentation import UNet_vanilla, ResDO_UNet


# =========================================================
# DRIVE DATASET
# =========================================================
class DriveDataset(Dataset):
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


# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser(description="URVSM Segmentation (No Translation)")

# Essential
parser.add_argument('--datapath', required=True, help='path to input image folder')
parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--note', type=str, default='experiment')

# Model
parser.add_argument('--seg_model', default='Resdounet',
                    choices=['unet_vanilla', 'Resdounet'])

parser.add_argument('--segnet_checkpoint_unet',
                    default='./ckpt/segmentation.pth')

parser.add_argument('--segnet_checkpoint_resdounet',
                    default='./ckpt/segmentation_resdo.pth')

# Output
parser.add_argument('--save_root', default='./result',
                    help='root directory to save results')


# =========================================================
# MAIN
# =========================================================
def main():
    args = parser.parse_args()

    # Device
    device = torch.device(
        f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    )
    print("Using device:", device)

    # Output directory
    save_dir = os.path.join(args.save_root, args.note, 'segmentation')
    os.makedirs(save_dir, exist_ok=True)

    # Dataset & Loader
    dataset = DriveDataset(args.datapath)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    print("Dataloader ready. Images:", len(dataset))

    # Load model
    if args.seg_model == 'unet_vanilla':
        seg_net = UNet_vanilla()
        seg_net.load_state_dict(
            torch.load(args.segnet_checkpoint_unet, map_location=device)
        )
    else:
        seg_net = ResDO_UNet(in_ch=4, out_ch=1)
        seg_net.load_state_dict(
            torch.load(args.segnet_checkpoint_resdounet, map_location=device)
        )

    seg_net = seg_net.to(device)
    seg_net.eval()

    # Inference
    with torch.no_grad():
        for img, name in loader:
            print("Processing:", name[0])

            img = img.to(device)
            N, C, H, W = img.shape
            assert N == 1

            # Initial empty segmentation
            init_seg = torch.zeros((N, 1, H, W), device=device)

            # URVSM iterative refinement (NO translation)
            x = torch.cat((init_seg, img), dim=1)
            y = seg_net(x)

            for _ in range(1):
                x = torch.cat((y, img), dim=1)
                y = seg_net(x)

            out = y[0, 0].cpu().numpy()
            out = (out * 255).astype(np.uint8)

            cv.imwrite(os.path.join(save_dir, name[0]), out)

    print("Segmentation completed successfully!")


if __name__ == '__main__':
    main()
