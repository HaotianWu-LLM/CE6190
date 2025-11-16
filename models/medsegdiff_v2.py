"""
MedSegDiff-V2: Diffusion-based Medical Image Segmentation with Transformer
Based on: Enhanced version with Vision Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from timm.models.vision_transformer import VisionTransformer
import numpy as np


class TransformerEncoder(nn.Module):
    """Vision Transformer encoder for feature extraction"""

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(6)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/16, W/16]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoding
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class TransformerConditionalUNet(nn.Module):
    """UNet with Transformer encoder for conditioning"""

    def __init__(self, in_channels=3, cond_channels=1, image_size=256):
        super().__init__()

        self.image_size = image_size

        # Transformer encoder for image features
        self.transformer = TransformerEncoder(
            img_size=image_size,
            patch_size=16,
            in_chans=in_channels,
            embed_dim=512
        )

        # Projection to spatial features
        self.feature_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 64)
        )

        # Diffusion UNet
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=cond_channels + 64,  # Noisy mask + transformer features
            out_channels=cond_channels,
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
        # Extract transformer features
        trans_feat = self.transformer(condition)  # [B, num_patches, 512]
        trans_feat = self.feature_proj(trans_feat)  # [B, num_patches, 64]

        # Reshape to spatial
        B = trans_feat.shape[0]
        H = W = self.image_size // 16
        trans_feat = trans_feat.transpose(1, 2).reshape(B, 64, H, W)

        # Upsample to match input size
        trans_feat = F.interpolate(
            trans_feat, size=(self.image_size, self.image_size),
            mode='bilinear', align_corners=False
        )

        # Concatenate with noisy mask
        x_cond = torch.cat([x, trans_feat], dim=1)

        return self.unet(x_cond, timestep).sample


class MedSegDiffV2(nn.Module):
    """
    MedSegDiff-V2: Enhanced with Vision Transformer
    """

    def __init__(self, in_channels=3, num_classes=1, image_size=256,
                 num_train_timesteps=1000, beta_schedule="linear",
                 num_inference_steps=50):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps

        # Transformer-based conditional UNet
        self.model = TransformerConditionalUNet(
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
        """Generate segmentation masks"""
        batch_size = images.shape[0]
        device = images.device

        # Start from random noise
        mask = torch.randn(
            batch_size, self.num_classes, self.image_size, self.image_size,
            device=device
        )

        # Set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                noise_pred = self.model(
                    mask,
                    timestep=t.to(device).unsqueeze(0),
                    condition=images
                )

            mask = self.scheduler.step(noise_pred, t, mask).prev_sample

        return mask

    def training_step(self, images, masks):
        """Training step"""
        batch_size = masks.shape[0]
        device = masks.device

        # Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()

        # Add noise
        noise = torch.randn_like(masks)
        noisy_masks = self.scheduler.add_noise(masks, noise, timesteps)

        # Predict noise
        noise_pred = self.model(noisy_masks, timesteps, images)

        # Loss
        diffusion_loss = F.mse_loss(noise_pred, noise)

        return {
            'loss': diffusion_loss,
            'diffusion_loss': diffusion_loss
        }

    def _dice_coefficient(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class MedSegDiffV2Trainer:
    """Trainer for MedSegDiff-V2"""

    def __init__(self, model, optimizer, device='cuda', **kwargs):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()

    def train_epoch(self, train_loader, epoch):
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
        self.model.eval()
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)

            output = self.model(image)
            pred = torch.sigmoid(output)

        return pred.cpu().numpy()