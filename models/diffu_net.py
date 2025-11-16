"""
Diff-UNet: A Diffusion Embedded Network for Volumetric Segmentation
Based on: https://arxiv.org/abs/2303.10326
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np


class DiffUNet(nn.Module):
    """
    Diff-UNet: Combines UNet architecture with diffusion process for segmentation
    """

    def __init__(self, in_channels=3, num_classes=1, image_size=256,
                 num_train_timesteps=1000, beta_schedule="linear"):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size

        # Encoder (uses diffusion UNet as feature extractor)
        self.encoder = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=in_channels,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        # Diffusion scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule
        )

    def forward(self, x, return_features=False):
        """
        Forward pass for inference
        Args:
            x: Input image [B, C, H, W]
            return_features: Whether to return intermediate features
        """
        # Denoise through diffusion process
        denoised = self._denoise_image(x)

        # Generate segmentation
        seg_logits = self.seg_head(denoised)

        if return_features:
            return seg_logits, denoised
        return seg_logits

    def _denoise_image(self, x):
        """Apply diffusion denoising to input image"""
        # Add small noise and denoise (simplified inference)
        noise_level = 0.1
        noise = torch.randn_like(x) * noise_level
        noisy_x = x + noise

        # Single denoising step through encoder
        denoised = self.encoder(noisy_x, timestep=torch.tensor([500]).to(x.device)).sample
        return denoised

    def training_step(self, images, masks):
        """
        Training step combining diffusion and segmentation loss
        Args:
            images: Input images [B, C, H, W]
            masks: Ground truth masks [B, 1, H, W]
        """
        batch_size = images.shape[0]

        # 1. Diffusion loss - train encoder to denoise
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=images.device
        ).long()

        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        noise_pred = self.encoder(noisy_images, timesteps).sample
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # 2. Segmentation loss
        seg_logits = self.forward(images)
        seg_loss = F.binary_cross_entropy_with_logits(seg_logits, masks)

        # Dice loss
        seg_probs = torch.sigmoid(seg_logits)
        dice_loss = 1 - self._dice_coefficient(seg_probs, masks)

        # Combined loss
        total_loss = diffusion_loss + seg_loss + dice_loss

        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'seg_loss': seg_loss,
            'dice_loss': dice_loss
        }

    def _dice_coefficient(self, pred, target, smooth=1e-6):
        """Calculate Dice coefficient"""
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class DiffUNetTrainer:
    """Trainer class for Diff-UNet"""

    def __init__(self, model, optimizer, device='cuda', **kwargs):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                losses = self.model.training_step(images, masks)
                loss = losses['loss']

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                preds = torch.sigmoid(outputs) > 0.5

                dice = self.model._dice_coefficient(preds.float(), masks)
                total_dice += dice.item()

        return total_dice / len(val_loader)

    def predict(self, image):
        """Predict segmentation for single image"""
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

            output = self.model(image)
            pred = torch.sigmoid(output)

        return pred.cpu().numpy()