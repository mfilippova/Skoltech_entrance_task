from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from minDiffusion.mindiffusion.unet import NaiveUnet
from minDiffusion.mindiffusion.ddpm import DDPM

from PIL import Image
import glob


class TrainData(Dataset):  # добавила класс для чтения и обработки данных
    def __init__(self):
        self.data = glob.glob('resized_maps_train/*.jpg')
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.data[idx])
        image = self.tf(image)

        return image


def resize_dataset():  # добавила функцию для сжатия размера картинок в датасете
    tf = transforms.Compose([
        transforms.Resize((64, 64)),
    ])
    for path in tqdm(glob.glob('train-org-img/*.jpg')):
        image = Image.open(path)
        image = tf(image)
        image.save(f"resized_maps_train/{path.split('/')[1]}")


def train_maps(
    start_epoch: int = 0,
    n_epoch: int = 100,
    device: str = "cuda:1",
    load_pth: Optional[str] = None
) -> None:

    ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128), betas=(1e-4, 0.02), n_T=1000)

    if load_pth is not None:
        ddpm.load_state_dict(torch.load(load_pth))

    ddpm.to(device)

    dataset = TrainData()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-7)
    sheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.1)  # добавила lr sheduler каждые 50 эпох

    for i in range(start_epoch, n_epoch):
        print(f"Epoch {i} : ")
        ddpm.train()

        pbar = tqdm(dataloader)
        loss_ema = None
        for x in pbar:
            optim.zero_grad()
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        sheduler.step()

        ddpm.eval()
        with torch.no_grad():
            xh = ddpm.sample(8, (3, 64, 64), device)
            xset = torch.cat([xh, x[:8]], dim=0)
            grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
            save_image(grid, f"images/maps{i:03d}.png")

            # save model
            if (i + 1) % 10 == 0:
                torch.save(ddpm.state_dict(), f"weights/maps.pth")


if __name__ == "__main__":
    resize_dataset()
    train_maps(start_epoch=0, n_epoch=300, device="cuda:0", load_pth=None)
