"""
SegGuidedDiff: Anatomically-Controllable Medical Image Generation
with Segmentation-Guided Diffusion Models
Based on: https://arxiv.org/abs/2402.05210
Adapted for segmentation task using bidirectional conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
import numpy as np


class SegGuidedDiff(nn.Module):
    """
    Segmentation-Guided Diffusion adapted for segmentation task
    Uses bidirectional conditioning: image -> mask and mask -> image
    """

    def __init__(self, in_channels=3, num_classes=1, image_size=256,
                 num_train_timesteps=1000, beta_schedule="linear",
                 num_inference_steps=50, use_ddim=True, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps

        # Image-to-Mask UNet (for segmentation)
        self.img2mask_unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels + num_classes,  # Image + noisy mask
            out_channels=num_classes,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
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
                "UpBlock2D",
                "UpBlock2D"
            ),
        )

        # Mask-to-Image UNet (for consistency)
        self.mask2img_unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels + num_classes,  # Noisy image + mask
            out_channels=in_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
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
                "UpBlock2D",
                "UpBlock2D"
            ),
        )

        # Schedulers
        if use_ddim:
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule
            )

    def forward(self, images):
        """
        Generate segmentation from images
        Args:
            images: Input images [B, C, H, W]
        Returns:
            Predicted masks [B, 1, H, W]
        """
        batch_size = images.shape[0]
        device = images.device

        # Start from random noise for mask
        mask = torch.randn(
            batch_size, self.num_classes, self.image_size, self.image_size,
            device=device
        )

        # Set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising conditioned on image
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                # Concatenate image as condition
                model_input = torch.cat([images, mask], dim=1)
                noise_pred = self.img2mask_unet(model_input, t).sample

            # Denoise step
            mask = self.scheduler.step(noise_pred, t, mask).prev_sample

        return mask

    def training_step(self, images, masks):
        """
        Bidirectional training: image->mask and mask->image
        """
        batch_size = masks.shape[0]
        device = masks.device

        # Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()

        # 1. Train image-to-mask path
        noise_mask = torch.randn_like(masks)
        noisy_masks = self.scheduler.add_noise(masks, noise_mask, timesteps)

        model_input = torch.cat([images, noisy_masks], dim=1)
        noise_pred_mask = self.img2mask_unet(model_input, timesteps).sample

        loss_img2mask = F.mse_loss(noise_pred_mask, noise_mask)

        # 2. Train mask-to-image path (for consistency)
        noise_img = torch.randn_like(images)
        noisy_images = self.scheduler.add_noise(images, noise_img, timesteps)

        model_input = torch.cat([noisy_images, masks], dim=1)
        noise_pred_img = self.mask2img_unet(model_input, timesteps).sample

        loss_mask2img = F.mse_loss(noise_pred_img, noise_img)

        # 3. Segmentation consistency loss (sparse, for stability)
        if torch.rand(1).item() < 0.1:  # 10% of the time
            with torch.no_grad():
                pred_masks = self.forward(images)

            seg_loss = F.binary_cross_entropy_with_logits(pred_masks, masks)
            dice_loss = 1 - self._dice_coefficient(torch.sigmoid(pred_masks), masks)
            total_loss = loss_img2mask + 0.5 * loss_mask2img + 0.1 * (seg_loss + dice_loss)
        else:
            seg_loss = torch.tensor(0.0)
            dice_loss = torch.tensor(0.0)
            total_loss = loss_img2mask + 0.5 * loss_mask2img

        return {
            'loss': total_loss,
            'img2mask_loss': loss_img2mask,
            'mask2img_loss': loss_mask2img,
            'seg_loss': seg_loss,
            'dice_loss': dice_loss
        }

    def _dice_coefficient(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class SegGuidedDiffTrainer:
    """Trainer for SegGuidedDiff"""

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