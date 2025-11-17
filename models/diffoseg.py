"""
DiffOSeg: Omni Medical Image Segmentation via Multi-Expert Collaboration Diffusion Model
Based on: https://arxiv.org/abs/2507.13087
Note: This implementation adapts the multi-expert collaboration concept for single-annotation segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
import numpy as np


class ExpertUNet(nn.Module):
    """Single expert UNet for specific feature extraction"""

    def __init__(self, in_channels, out_channels, image_size=256, expert_id=0):
        super().__init__()

        self.expert_id = expert_id

        # Specialized channels for each expert
        # Use multiples of 32 to ensure compatibility with GroupNorm
        base_channels = [64, 128, 256, 512]
        # Add 32 * expert_id to ensure all channels are divisible by 32
        expert_channels = [c + expert_id * 32 for c in base_channels]

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=tuple(expert_channels),
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

    def forward(self, x, timestep):
        return self.unet(x, timestep).sample


class ExpertCollaborationModule(nn.Module):
    """Module to fuse multiple expert predictions"""

    def __init__(self, num_experts, feature_dim):
        super().__init__()

        self.num_experts = num_experts

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_experts) / num_experts)

        # Refinement network
        self.refine = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(num_groups=min(32, feature_dim), num_channels=feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1)
        )

    def forward(self, expert_features):
        """
        Args:
            expert_features: List of [B, C, H, W] from each expert
        Returns:
            Fused features [B, C, H, W]
        """
        # Weighted fusion with softmax normalized weights
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted = sum(w * feat for w, feat in zip(weights, expert_features))

        # Refine fused features
        refined = self.refine(weighted)

        return refined


class DiffOSeg(nn.Module):
    """
    DiffOSeg: Multi-Expert Collaboration Diffusion Model
    """

    def __init__(self, in_channels=3, num_classes=1, image_size=256,
                 num_experts=3, num_train_timesteps=1000,
                 beta_schedule="linear", num_inference_steps=50, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_experts = num_experts
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps

        # Multiple expert networks
        self.experts = nn.ModuleList([
            ExpertUNet(
                in_channels=in_channels + num_classes,
                out_channels=num_classes,
                image_size=image_size,
                expert_id=i
            )
            for i in range(num_experts)
        ])

        # Collaboration module
        self.collaboration = ExpertCollaborationModule(
            num_experts=num_experts,
            feature_dim=num_classes
        )

        # Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule
        )

        # Final segmentation head
        self.seg_head = nn.Conv2d(num_classes, num_classes, 1)

    def forward(self, images):
        """
        Generate segmentation using multi-expert collaboration
        """
        batch_size = images.shape[0]
        device = images.device

        # Start from random noise
        mask = torch.randn(
            batch_size, self.num_classes, self.image_size, self.image_size,
            device=device
        )

        # Set timesteps
        self.scheduler.set_timesteps(self.num_inference_steps)

        # Iterative denoising with expert collaboration
        for t in self.scheduler.timesteps:
            with torch.no_grad():
                # Get predictions from all experts
                model_input = torch.cat([images, mask], dim=1)

                expert_preds = []
                for expert in self.experts:
                    pred = expert(model_input, t.to(device).unsqueeze(0))
                    expert_preds.append(pred)

                # Collaborate and fuse
                noise_pred = self.collaboration(expert_preds)

            # Denoise step
            mask = self.scheduler.step(noise_pred, t, mask).prev_sample

        # Final refinement
        mask = self.seg_head(mask)

        return mask

    def training_step(self, images, masks):
        """
        Train with expert specialization and collaboration
        """
        batch_size = masks.shape[0]
        device = masks.device

        # Sample timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()

        # Add noise to masks
        noise = torch.randn_like(masks)
        noisy_masks = self.scheduler.add_noise(masks, noise, timesteps)

        # Get predictions from all experts
        model_input = torch.cat([images, noisy_masks], dim=1)

        expert_preds = []
        expert_losses = []

        for expert in self.experts:
            pred = expert(model_input, timesteps)
            expert_preds.append(pred)

            # Individual expert loss
            expert_loss = F.mse_loss(pred, noise)
            expert_losses.append(expert_loss)

        # Collaboration loss
        collab_pred = self.collaboration(expert_preds)
        collab_loss = F.mse_loss(collab_pred, noise)

        # Diversity loss (encourage expert specialization)
        # Negative MSE encourages diversity
        diversity_loss = 0
        num_pairs = 0
        for i in range(len(expert_preds)):
            for j in range(i + 1, len(expert_preds)):
                diversity_loss -= F.mse_loss(expert_preds[i], expert_preds[j])
                num_pairs += 1

        if num_pairs > 0:
            diversity_loss = diversity_loss / num_pairs

        # Segmentation consistency (sparse, for stability)
        if torch.rand(1).item() < 0.1:  # 10% of the time
            with torch.no_grad():
                pred_masks = self.forward(images)

            seg_loss = F.binary_cross_entropy_with_logits(pred_masks, masks)
            dice_loss = 1 - self._dice_coefficient(torch.sigmoid(pred_masks), masks)

            total_loss = (
                collab_loss +
                0.3 * sum(expert_losses) / len(expert_losses) +
                0.1 * diversity_loss +
                0.1 * (seg_loss + dice_loss)
            )
        else:
            seg_loss = torch.tensor(0.0, device=device)
            dice_loss = torch.tensor(0.0, device=device)

            total_loss = (
                collab_loss +
                0.3 * sum(expert_losses) / len(expert_losses) +
                0.1 * diversity_loss
            )

        return {
            'loss': total_loss,
            'collab_loss': collab_loss,
            'expert_loss': sum(expert_losses) / len(expert_losses),
            'diversity_loss': diversity_loss,
            'seg_loss': seg_loss,
            'dice_loss': dice_loss
        }

    def _dice_coefficient(self, pred, target, smooth=1e-6):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class DiffOSegTrainer:
    """Trainer for DiffOSeg"""

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