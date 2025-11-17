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
        elif self.dataset_name == 'glas':
            self.prepare_glas()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def prepare_isic2016(self):
        """Prepare ISIC 2016 Skin Lesion Segmentation Dataset"""
        print("Preparing ISIC 2016 dataset...")

        # Create directories
        for split in ['train', 'test']:
            (self.dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

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

    def prepare_glas(self):
        """Prepare GlaS - Gland Segmentation in Colon Histology Images Dataset"""
        print("Preparing GlaS dataset...")

        # Create directories
        for split in ['train', 'test']:
            (self.dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

        print("Note: Please manually download GlaS from https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/")
        print("Expected directory structure (original data):")
        print(f"{self.dataset_dir}/")
        print("  ├── train/")
        print("  │   ├── train_1.bmp")
        print("  │   ├── train_1_anno.bmp")
        print("  │   ├── train_2.bmp")
        print("  │   └── ...")
        print("  └── test/  (testA and testB files mixed)")
        print("      ├── testA_1.bmp")
        print("      ├── testA_1_anno.bmp")
        print("      ├── testB_1.bmp")
        print("      ├── testB_1_anno.bmp")
        print("      └── ...")

        # Process images
        self._process_glas_split('train', 'train')
        self._process_glas_split('test', 'test')

        print(f"GlaS dataset prepared at {self.dataset_dir}")

    def _process_glas_split(self, source_split, target_split):
        """Process GlaS images and masks"""
        # Handle different possible source directory structures
        possible_sources = [
            self.dataset_dir / source_split,
            self.dataset_dir / 'Warwick QU Dataset (Released 2016_07_08)' / source_split,
            self.dataset_dir / 'GlaS' / source_split,
        ]

        img_source = None
        for src in possible_sources:
            if src.exists():
                img_source = src
                break

        if img_source is None:
            print(f"Warning: Source directory for {source_split} split not found. Skipping...")
            print(f"Tried the following paths:")
            for src in possible_sources:
                print(f"  - {src}")
            return

        img_dest = self.dataset_dir / target_split / 'images'
        mask_dest = self.dataset_dir / target_split / 'masks'

        # Find all image files (not annotation files)
        # For train: train_*.bmp (excluding *_anno.bmp)
        # For test: testA_*.bmp and testB_*.bmp (excluding *_anno.bmp)
        all_bmp_files = sorted(list(img_source.glob('*.bmp')))

        # Filter out annotation files
        img_files = [f for f in all_bmp_files if '_anno' not in f.stem]

        if len(img_files) == 0:
            print(f"Warning: No image files found in {img_source}. Skipping...")
            return

        print(f"Found {len(img_files)} images in {source_split} split")

        for img_path in tqdm(img_files, desc=f"Processing {source_split} split"):
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')

                # Resize to 512x512 (GlaS images are typically 775x522)
                img_resized = img.resize((512, 512), Image.BILINEAR)

                # Save as PNG (better for medical images)
                output_name = img_path.stem + '.png'
                img_resized.save(img_dest / output_name)

                # Process corresponding mask
                # Mask naming: original_name_anno.bmp
                mask_name = img_path.stem + '_anno.bmp'
                mask_path = img_source / mask_name

                if mask_path.exists():
                    mask = Image.open(mask_path).convert('L')
                    mask_resized = mask.resize((512, 512), Image.NEAREST)

                    # Binarize mask (GlaS uses different values for different glands)
                    # We convert to binary: any non-zero value becomes 255
                    mask_np = np.array(mask_resized)
                    mask_np = (mask_np > 0).astype(np.uint8) * 255

                    Image.fromarray(mask_np).save(mask_dest / output_name)
                else:
                    print(f"Warning: Mask not found for {img_path.name}: {mask_path}")

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

        print(f"Processed {len(list(img_dest.glob('*.png')))} images for {target_split} split")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare medical image datasets')
    parser.add_argument('--dataset', type=str, required=True, choices=['isic2016', 'glas'],
                        help='Dataset to prepare')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Root directory for datasets')

    args = parser.parse_args()

    preparator = DatasetPreparator(args.dataset, args.data_dir)
    preparator.prepare()