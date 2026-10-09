"""
Fine-Tuning Loop for Anime Domain Adaptation (Pillar 2).

Fine-tunes the APISR RRDB-6B generator on synthetic broadcast degradation pairs
extracted from the Re-Anime600 dataset.

Loss Function:
  L_total = L_1 + 0.5 * L_edge
  where L_edge is a differentiable 2D Sobel gradient loss penalizing line art blur.

Hardware:
  Optimized for NVIDIA GeForce RTX 4060 (8GB VRAM) using torch.amp mixed precision.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from apisr_arch import RRDBNet
from src.dataset import AnimePatchDataset


class AnimeEdgeLoss(nn.Module):
    """Differentiable 2D Sobel edge loss to enforce razor-sharp anime ink contours."""

    def __init__(self, device: torch.device):
        super().__init__()
        # Sobel horizontal and vertical kernels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device).view(1, 1, 3, 3)

        # Repeat for 3 color channels with groups=3
        self.kernel_x = sobel_x.repeat(3, 1, 1, 1)
        self.kernel_y = sobel_y.repeat(3, 1, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Pad with reflection to keep exact dimensions
        pred_pad = F.pad(pred, (1, 1, 1, 1), mode="reflect")
        target_pad = F.pad(target, (1, 1, 1, 1), mode="reflect")

        pred_grad_x = F.conv2d(pred_pad, self.kernel_x, groups=3)
        pred_grad_y = F.conv2d(pred_pad, self.kernel_y, groups=3)
        target_grad_x = F.conv2d(target_pad, self.kernel_x, groups=3)
        target_grad_y = F.conv2d(target_pad, self.kernel_y, groups=3)

        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        return loss_x + loss_y


def train_domain_adaptation(
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    samples_per_epoch: int = 200,
    pretrained_weights: str = "model_zoo/2x_APISR_RRDB_GAN_generator.pth",
    save_dir: str = "checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("AIAnime — Pillar 2 Domain Adaptation Fine-Tuning")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 65)

    os.makedirs(save_dir, exist_ok=True)

    # 1. Initialize Generator Model
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=2,
        num_feat=64,
        num_block=6,
        num_grow_ch=32,
    ).to(device)

    # Load baseline pretrained APISR weights
    if os.path.isfile(pretrained_weights):
        ckpt = torch.load(pretrained_weights, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "params_ema" in ckpt:
            state_dict = ckpt["params_ema"]
        elif isinstance(ckpt, dict) and "params" in ckpt:
            state_dict = ckpt["params"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded pretrained backbone weights from {pretrained_weights}")
    else:
        print(f"[WARNING] Pretrained weights not found at {pretrained_weights}. Starting from random initialization.")

    model.train()

    # 2. Dataset and Dataloader
    dataset = AnimePatchDataset(
        data_dir="data/train/train",
        hr_patch_size=256,
        scale=2,
        samples_per_epoch=samples_per_epoch,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 3. Optimizers & Losses
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader), eta_min=1e-6)

    criterion_l1 = nn.L1Loss()
    criterion_edge = AnimeEdgeLoss(device=device)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    print(f"Training Config:")
    print(f"  Epochs:             {epochs}")
    print(f"  Batch Size:         {batch_size}")
    print(f"  Samples per Epoch:  {samples_per_epoch}")
    print(f"  Steps per Epoch:    {len(dataloader)}")
    print(f"  Peak Precision:     AMP float16")
    print("-" * 65)

    best_loss = float("inf")

    # 4. Training Loop
    start_total = time.time()
    for epoch in range(1, epochs + 1):
        epoch_l1 = 0.0
        epoch_edge = 0.0
        epoch_total = 0.0
        start_epoch = time.time()

        for step, batch in enumerate(dataloader, start=1):
            lr_imgs = batch["lr"].to(device, non_blocking=True)
            hr_imgs = batch["hr"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                pred_hr = model(lr_imgs)
                l1_loss = criterion_l1(pred_hr, hr_imgs)
                edge_loss = criterion_edge(pred_hr, hr_imgs)
                total_loss = l1_loss + 0.5 * edge_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_l1 += l1_loss.item()
            epoch_edge += edge_loss.item()
            epoch_total += total_loss.item()

            if step % 10 == 0 or step == len(dataloader):
                print(
                    f"Epoch [{epoch}/{epochs}] Step [{step}/{len(dataloader)}] "
                    f"L1: {l1_loss.item():.4f} | Edge: {edge_loss.item():.4f} | Total: {total_loss.item():.4f}"
                )

        num_steps = len(dataloader)
        avg_loss = epoch_total / num_steps
        epoch_sec = time.time() - start_epoch
        print(f"--> Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_sec:.1f}s")

        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"apisr_reanime600_epoch{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "apisr_reanime600_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"    * New best model saved: {best_path} (Loss: {best_loss:.4f})")

    total_sec = time.time() - start_total
    print("=" * 65)
    print(f"Training Finished in {total_sec:.1f}s ({total_sec / 60:.2f} min).")
    print(f"Best Weights: {os.path.join(save_dir, 'apisr_reanime600_best.pth')}")
    print("=" * 65)


if __name__ == "__main__":
    train_domain_adaptation()
