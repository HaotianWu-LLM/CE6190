"""
MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model
Based on: https://arxiv.org/abs/2211.00611
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np


class ConditionalUNet(nn.Module):
    """UNet conditioned on segmentation masks"""

    def __init__(self, in_channels=3, cond_channels=1, image_size=256):
        super().__init__()

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels + cond_channels,  # Image + mask condition
            out_channels=cond_channels,  # Predict mask
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

    def forward(self, x, timestep, condition):
        """
        Args:
            x: Noisy mask [B, 1, H, W]
            timestep: Diffusion timestep
            condition: Conditioning image [B, C, H, W]
        """
        # Concatenate condition
        x_cond = torch.cat([x, condition], dim=1)
        return self.unet(x_cond, timestep).sample


class MedSegDiff(nn.Module):
    """
    MedSegDiff: Conditional diffusion model for medical image segmentation
    """

    def __init__(self, in_channels=3, num_classes=1, image_size=256,
                 num_train_timesteps=1000, beta_schedule="linear",
                 num_inference_steps=50):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps

        # Conditional UNet for mask generation
        self.model = ConditionalUNet(
            in_channels=in_channels,
            cond_channels=num_classes,
            image_size=image_size
        )

        # Noise scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule
        )

    def forward(self, images):
        """
        Generate segmentation masks from images using DDPM sampling
        Args:
            images: Input images [B, C, H, W]
        Returns:
            Predicted masks [B, 1, H, W]
        """
        batch_size = images.shape[0]
        device = images.device

        # Start from random noise
        mask = torch.randn(
            batch_size, self.num_classes, self.image_size, self.image_size,
            device=device
        )

        # Set timesteps for inference
        self.scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising
        for t in self.scheduler.timesteps:
            # Predict noise
            with torch.no_grad():
                noise_pred = self.model(
                    mask,
                    timestep=t.to(device).unsqueeze(0),
                    condition=images
                )

            # Denoise step
            mask = self.scheduler.step(
                noise_pred, t, mask
            ).prev_sample

        return mask

    def training_step(self, images, masks):
        """
        Training step for MedSegDiff
        Args:
            images: Input images [B, C, H, W]
            masks: Ground truth masks [B, 1, H, W]
        """
        batch_size = masks.shape[0]
        device = masks.device

        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()

        # Add noise to masks
        noise = torch.randn_like(masks)
        noisy_masks = self.scheduler.add_noise(masks, noise, timesteps)

        # Predict noise conditioned on images
        noise_pred = self.model(noisy_masks, timesteps, images)

        # Diffusion loss
        diffusion_loss = F.mse_loss(noise_pred, noise)

        # Optional: Add segmentation consistency loss
        # Generate prediction and compare with GT
        with torch.no_grad():
            pred_masks = self.forward(images)

        seg_loss = F.binary_cross_entropy_with_logits(pred_masks, masks)
        dice_loss = 1 - self._dice_coefficient(torch.sigmoid(pred_masks), masks)

        # Weighted combination
        total_loss = diffusion_loss + 0.1 * (seg_loss + dice_loss)

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


class MedSegDiffTrainer:
    """Trainer for MedSegDiff"""

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
        """Predict segmentation"""
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

            output = self.model(image)
            pred = torch.sigmoid(output)

        return pred.cpu().numpy()