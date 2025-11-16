"""
Main script for training and evaluating diffusion-based segmentation models
Supports: Diff-UNet, MedSegDiff, MedSegDiff-V2, SegGuidedDiff, DiffOSeg
Datasets: ISIC 2016, DRIVE
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# Import models
from models.diffu_net import DiffUNet, DiffUNetTrainer
from models.medsegdiff import MedSegDiff, MedSegDiffTrainer
from models.medsegdiff_v2 import MedSegDiffV2, MedSegDiffV2Trainer
from models.segguideddiff import SegGuidedDiff, SegGuidedDiffTrainer
from models.diffoseg import DiffOSeg, DiffOSegTrainer

# Import metrics
from metrics import MetricsTracker, SegmentationMetrics

# ==================== HYPERPARAMETERS ====================
CONFIGS = {
    'isic2016': {
        'image_size': 256,
        'in_channels': 3,
        'num_classes': 1,
        'batch_size': 16,
        'lr': 1e-4,
        'epochs': 200,
        'num_inference_steps': 50,
    },
    'drive': {
        'image_size': 512,
        'in_channels': 3,
        'num_classes': 1,
        'batch_size': 8,
        'lr': 1e-4,
        'epochs': 200,
        'num_inference_steps': 50,
    }
}

MODEL_SPECIFIC_CONFIGS = {
    'diffu_net': {
        'num_train_timesteps': 1000,
        'beta_schedule': 'linear'
    },
    'medsegdiff': {
        'num_train_timesteps': 1000,
        'beta_schedule': 'linear'
    },
    'medsegdiff_v2': {
        'num_train_timesteps': 1000,
        'beta_schedule': 'linear'
    },
    'segguideddiff': {
        'num_train_timesteps': 1000,
        'beta_schedule': 'linear',
        'use_ddim': True
    },
    'diffoseg': {
        'num_train_timesteps': 1000,
        'beta_schedule': 'linear',
        'num_experts': 3
    }
}


# ==================== DATASET ====================
class MedicalSegmentationDataset(Dataset):
    """Generic medical image segmentation dataset"""

    def __init__(self, data_dir, split='train', transform=None, target_transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.target_transform = target_transform

        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'

        self.images = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / (img_path.stem + '.png')

        # Load image
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def get_transforms(image_size):
    """Get image and mask transforms"""

    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    return image_transform, mask_transform


def get_dataloaders(dataset_name, data_dir, batch_size, image_size):
    """Create train and test dataloaders"""

    image_transform, mask_transform = get_transforms(image_size)

    train_dataset = MedicalSegmentationDataset(
        data_dir=data_dir,
        split='train',
        transform=image_transform,
        target_transform=mask_transform
    )

    test_dataset = MedicalSegmentationDataset(
        data_dir=data_dir,
        split='test',
        transform=image_transform,
        target_transform=mask_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader


# ==================== MODEL FACTORY ====================
def create_model(model_name, config):
    """Create model instance"""

    base_config = {
        'in_channels': config['in_channels'],
        'num_classes': config['num_classes'],
        'image_size': config['image_size'],
        'num_inference_steps': config['num_inference_steps']
    }

    # Add model-specific configs
    model_config = {**base_config, **MODEL_SPECIFIC_CONFIGS[model_name]}

    if model_name == 'diffu_net':
        return DiffUNet(**model_config)
    elif model_name == 'medsegdiff':
        return MedSegDiff(**model_config)
    elif model_name == 'medsegdiff_v2':
        return MedSegDiffV2(**model_config)
    elif model_name == 'segguideddiff':
        return SegGuidedDiff(**model_config)
    elif model_name == 'diffoseg':
        return DiffOSeg(**model_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_trainer(model_name, model, optimizer, device):
    """Create trainer instance"""

    if model_name == 'diffu_net':
        return DiffUNetTrainer(model, optimizer, device)
    elif model_name == 'medsegdiff':
        return MedSegDiffTrainer(model, optimizer, device)
    elif model_name == 'medsegdiff_v2':
        return MedSegDiffV2Trainer(model, optimizer, device)
    elif model_name == 'segguideddiff':
        return SegGuidedDiffTrainer(model, optimizer, device)
    elif model_name == 'diffoseg':
        return DiffOSegTrainer(model, optimizer, device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ==================== TRAINING ====================
def train_model(model_name, dataset_name, data_dir, args):
    """Train a single model"""

    print(f"\n{'=' * 60}")
    print(f"Training {model_name} on {dataset_name}")
    print(f"{'=' * 60}\n")

    # Get config
    config = CONFIGS[dataset_name].copy()
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['epochs'] = args.epochs
    if args.lr:
        config['lr'] = args.lr

    # Create dataloaders
    train_loader, test_loader = get_dataloaders(
        dataset_name,
        data_dir,
        config['batch_size'],
        config['image_size']
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name, config)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])

    # Create trainer
    trainer = create_trainer(model_name, model, optimizer, device)

    # Create checkpoint directory
    checkpoint_dir = Path(args.results_dir) / dataset_name / model_name / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_dice = 0
    train_losses = []
    val_dices = []

    for epoch in range(config['epochs']):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)
        train_losses.append(train_loss)

        # Validate every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_dice = trainer.validate(test_loader)
            val_dices.append(val_dice)

            print(f"Epoch {epoch + 1}/{config['epochs']}: "
                  f"Train Loss = {train_loss:.4f}, Val Dice = {val_dice:.4f}")

            # Save best model
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    model.state_dict(),
                    checkpoint_dir / 'best_model.pth'
                )
                print(f"Saved best model with Dice = {best_dice:.4f}")
        else:
            print(f"Epoch {epoch + 1}/{config['epochs']}: Train Loss = {train_loss:.4f}")

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / 'final_model.pth')

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_dices': val_dices,
        'best_dice': best_dice
    }

    with open(checkpoint_dir.parent / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining completed! Best Dice: {best_dice:.4f}\n")


# ==================== EVALUATION ====================
def evaluate_model(model_name, dataset_name, data_dir, args):
    """Evaluate a single model"""

    print(f"\n{'=' * 60}")
    print(f"Evaluating {model_name} on {dataset_name}")
    print(f"{'=' * 60}\n")

    # Get config
    config = CONFIGS[dataset_name].copy()

    # Create dataloader
    _, test_loader = get_dataloaders(
        dataset_name,
        data_dir,
        config['batch_size'],
        config['image_size']
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name, config)

    # Load checkpoint
    checkpoint_path = Path(args.results_dir) / dataset_name / model_name / 'checkpoints' / 'best_model.pth'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    # Create metrics tracker
    metrics_tracker = MetricsTracker()

    # Evaluate
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            # Predict
            outputs = model(images)
            preds_prob = torch.sigmoid(outputs)
            preds = (preds_prob > 0.5).float()

            # Calculate metrics for each sample
            for i in range(preds.shape[0]):
                pred_np = preds[i, 0].cpu().numpy()
                mask_np = masks[i, 0].cpu().numpy()
                pred_prob_np = preds_prob[i, 0].cpu().numpy()

                metrics_tracker.update(pred_np, mask_np, pred_prob_np)

    # Get average metrics
    avg_metrics = metrics_tracker.get_average_metrics()

    # Print metrics
    metrics_tracker.print_metrics()

    # Save metrics
    results_dir = Path(args.results_dir) / dataset_name / model_name
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(avg_metrics, f, indent=4)

    return avg_metrics


# ==================== TESTING (Qualitative Results) ====================
def test_model(model_name, dataset_name, data_dir, args):
    """Generate qualitative results"""

    print(f"\n{'=' * 60}")
    print(f"Testing {model_name} on {dataset_name}")
    print(f"{'=' * 60}\n")

    # Get config
    config = CONFIGS[dataset_name].copy()

    # Create dataloader
    _, test_loader = get_dataloaders(
        dataset_name,
        data_dir,
        1,  # Batch size 1 for visualization
        config['image_size']
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(model_name, config)

    # Load checkpoint
    checkpoint_path = Path(args.results_dir) / dataset_name / model_name / 'checkpoints' / 'best_model.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    # Create output directory
    qualitative_dir = Path(args.results_dir) / dataset_name / model_name / 'qualitative'
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations for first N samples
    num_samples = min(20, len(test_loader))

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            if idx >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Predict
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # Denormalize image for visualization
            image_vis = images[0].cpu()
            image_vis = image_vis * 0.5 + 0.5
            image_vis = image_vis.permute(1, 2, 0).numpy()

            mask_vis = masks[0, 0].cpu().numpy()
            pred_vis = preds[0, 0].cpu().numpy()

            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image_vis)
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(mask_vis, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pred_vis, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(qualitative_dir / f'sample_{idx:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Saved {num_samples} qualitative results to {qualitative_dir}\n")


# ==================== COMPARISON ====================
def generate_comparison_table(dataset_name, args):
    """Generate comparison table for all models"""

    print(f"\n{'=' * 60}")
    print(f"Generating Comparison Table for {dataset_name}")
    print(f"{'=' * 60}\n")

    results_dir = Path(args.results_dir) / dataset_name

    # Collect results from all models
    all_results = []

    for model_name in ['diffu_net', 'medsegdiff', 'medsegdiff_v2', 'segguideddiff', 'diffoseg']:
        metrics_path = results_dir / model_name / 'metrics.json'

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            result = {'Model': model_name}
            for key, value in metrics.items():
                if not key.endswith('_std'):
                    std_key = f'{key}_std'
                    std_value = metrics.get(std_key, 0)
                    result[key.upper()] = f"{value:.4f} ± {std_value:.4f}"

            all_results.append(result)

    # Create DataFrame
    if all_results:
        df = pd.DataFrame(all_results)

        # Save to CSV
        csv_path = results_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)

        # Print table
        print(df.to_string(index=False))
        print(f"\nSaved comparison table to {csv_path}\n")
    else:
        print("No results found for comparison\n")


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation with Diffusion Models')

    # Mode
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'eval', 'test', 'compare'],
                        help='Mode: train/eval/test/compare')

    # Dataset
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['isic2016', 'drive'],
                        help='Dataset name')

    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root data directory')

    # Models
    parser.add_argument('--models', type=str, default='all',
                        help='Models to run (comma-separated or "all")')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: from config)')

    # Output
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results directory')
    parser.add_argument('--save_results', action='store_true',
                        help='Save quantitative and qualitative results')

    args = parser.parse_args()

    # Parse models
    if args.models == 'all':
        models = ['diffu_net', 'medsegdiff', 'medsegdiff_v2', 'segguideddiff', 'diffoseg']
    else:
        models = [m.strip() for m in args.models.split(',')]

    # Get data directory
    data_dir = Path(args.data_dir) / args.dataset

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        print("Please run data_preparation.py first.")
        return

    # Execute based on mode
    if args.mode == 'train':
        for model_name in models:
            train_model(model_name, args.dataset, data_dir, args)

    elif args.mode == 'eval':
        for model_name in models:
            evaluate_model(model_name, args.dataset, data_dir, args)

        if args.save_results:
            generate_comparison_table(args.dataset, args)

    elif args.mode == 'test':
        for model_name in models:
            test_model(model_name, args.dataset, data_dir, args)

    elif args.mode == 'compare':
        generate_comparison_table(args.dataset, args)

    print("\nAll tasks completed!")


if __name__ == '__main__':
    main()