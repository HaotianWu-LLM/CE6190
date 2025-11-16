import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import argparse


class DatasetPreparator:
    def __init__(self, dataset_name, data_dir='./data'):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.dataset_dir = self.data_dir / dataset_name

    def prepare(self):
        if self.dataset_name == 'isic2016':
            self.prepare_isic2016()
        elif self.dataset_name == 'drive':
            self.prepare_drive()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def prepare_isic2016(self):
        """Prepare ISIC 2016 Skin Lesion Segmentation Dataset"""
        print("Preparing ISIC 2016 dataset...")

        # Create directories
        for split in ['train', 'test']:
            (self.dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

        # URLs (Note: Replace with actual ISIC 2016 download links)
        urls = {
            'train_images': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip',
            'train_masks': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip',
            'test_images': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip',
            'test_masks': 'https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip'
        }

        print("Note: Please manually download ISIC 2016 from https://challenge.isic-archive.com/data/")
        print("Expected directory structure:")
        print(f"{self.dataset_dir}/")
        print("  ├── ISBI2016_ISIC_Part1_Training_Data/")
        print("  ├── ISBI2016_ISIC_Part1_Training_GroundTruth/")
        print("  ├── ISBI2016_ISIC_Part1_Test_Data/")
        print("  └── ISBI2016_ISIC_Part1_Test_GroundTruth/")

        # Process images
        self._process_isic_split('train')
        self._process_isic_split('test')

        print(f"ISIC 2016 dataset prepared at {self.dataset_dir}")

    def _process_isic_split(self, split):
        """Process ISIC 2016 images and masks"""
        img_source = self.dataset_dir / f"ISBI2016_ISIC_Part1_{'Training' if split == 'train' else 'Test'}_Data"
        mask_source = self.dataset_dir / f"ISBI2016_ISIC_Part1_{'Training' if split == 'train' else 'Test'}_GroundTruth"

        if not img_source.exists() or not mask_source.exists():
            print(f"Warning: Source directories for {split} split not found. Skipping...")
            return

        img_dest = self.dataset_dir / split / 'images'
        mask_dest = self.dataset_dir / split / 'masks'

        img_files = sorted(list(img_source.glob('*.jpg')) + list(img_source.glob('*.png')))

        for img_path in tqdm(img_files, desc=f"Processing {split} split"):
            # Copy and resize image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256), Image.BILINEAR)
            img.save(img_dest / img_path.name)

            # Copy and resize mask
            mask_name = img_path.stem + '_Segmentation.png'
            mask_path = mask_source / mask_name

            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((256, 256), Image.NEAREST)
                # Binarize mask
                mask_np = np.array(mask)
                mask_np = (mask_np > 127).astype(np.uint8) * 255
                Image.fromarray(mask_np).save(mask_dest / (img_path.stem + '.png'))

    def prepare_drive(self):
        """Prepare DRIVE Retinal Vessel Segmentation Dataset"""
        print("Preparing DRIVE dataset...")

        # Create directories
        for split in ['train', 'test']:
            (self.dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

        print("Note: Please manually download DRIVE from https://drive.grand-challenge.org/")
        print("Expected directory structure:")
        print(f"{self.dataset_dir}/")
        print("  ├── training/")
        print("  │   ├── images/")
        print("  │   └── 1st_manual/")
        print("  └── test/")
        print("      ├── images/")
        print("      └── 1st_manual/")

        # Process images
        self._process_drive_split('training', 'train')
        self._process_drive_split('test', 'test')

        print(f"DRIVE dataset prepared at {self.dataset_dir}")

    def _process_drive_split(self, source_split, target_split):
        """Process DRIVE images and masks"""
        img_source = self.dataset_dir / source_split / 'images'
        mask_source = self.dataset_dir / source_split / '1st_manual'

        if not img_source.exists() or not mask_source.exists():
            print(f"Warning: Source directories for {source_split} split not found. Skipping...")
            return

        img_dest = self.dataset_dir / target_split / 'images'
        mask_dest = self.dataset_dir / target_split / 'masks'

        img_files = sorted(list(img_source.glob('*.tif')) + list(img_source.glob('*.png')))

        for img_path in tqdm(img_files, desc=f"Processing {source_split} split"):
            # Load and preprocess image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Apply CLAHE to green channel for better contrast
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

            # Resize to 512x512
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            Image.fromarray(img).save(img_dest / (img_path.stem + '.png'))

            # Process mask
            mask_name = img_path.stem.replace('_training', '').replace('_test', '') + '_manual1.gif'
            mask_path = mask_source / mask_name

            if mask_path.exists():
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((512, 512), Image.NEAREST)
                mask_np = np.array(mask)
                mask_np = (mask_np > 127).astype(np.uint8) * 255
                Image.fromarray(mask_np).save(mask_dest / (img_path.stem + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare medical image datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['isic2016', 'drive'],
                        help='Dataset to prepare')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root directory for datasets')

    args = parser.parse_args()

    preparator = DatasetPreparator(args.dataset, args.data_dir)
    preparator.prepare()