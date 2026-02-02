#!/usr/bin/env python
"""
Download MSCOCO 2014 dataset (train, val, test)
This script downloads and extracts the MSCOCO 2014 images
"""

import os
import urllib.request
import zipfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    print(f"Downloading {url}...")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc='Extracting'):
            zip_ref.extract(file, extract_to)

def main():
    # Base directory for MSCOCO data
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'MSCOCO')
    os.makedirs(base_dir, exist_ok=True)

    # MSCOCO 2014 dataset URLs
    datasets = {
        'train2014': 'http://images.cocodataset.org/zips/train2014.zip',
        'val2014': 'http://images.cocodataset.org/zips/val2014.zip',
        'test2014': 'http://images.cocodataset.org/zips/test2014.zip'
    }

    # Also download annotations
    annotations = {
        'train_val_annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        'test_annotations': 'http://images.cocodataset.org/annotations/image_info_test2014.zip'
    }

    print("=" * 80)
    print("MSCOCO 2014 Dataset Download Script")
    print("=" * 80)
    print(f"Download location: {base_dir}")
    print()

    # Download and extract image datasets
    for dataset_name, url in datasets.items():
        zip_filename = f"{dataset_name}.zip"
        zip_path = os.path.join(base_dir, zip_filename)
        dataset_dir = os.path.join(base_dir, dataset_name)

        # Check if already exists
        if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
            print(f"[OK] {dataset_name} already exists, skipping...")
            continue

        # Download
        if not os.path.exists(zip_path):
            download_url(url, zip_path)
        else:
            print(f"[OK] {zip_filename} already downloaded")

        # Extract
        extract_zip(zip_path, base_dir)

        # Remove zip file to save space
        print(f"Removing {zip_filename}...")
        os.remove(zip_path)

        print(f"[OK] {dataset_name} complete!\n")

    # Download and extract annotations
    print("\nDownloading annotations...")
    for ann_name, url in annotations.items():
        zip_filename = f"{ann_name}.zip"
        zip_path = os.path.join(base_dir, zip_filename)

        # Check if annotations already exist
        annotations_dir = os.path.join(base_dir, 'annotations')
        if os.path.exists(annotations_dir) and len(os.listdir(annotations_dir)) > 0:
            print(f"[OK] Annotations already exist, skipping...")
            break

        # Download
        if not os.path.exists(zip_path):
            download_url(url, zip_path)
        else:
            print(f"[OK] {zip_filename} already downloaded")

        # Extract
        extract_zip(zip_path, base_dir)

        # Remove zip file
        print(f"Removing {zip_filename}...")
        os.remove(zip_path)

        print(f"[OK] {ann_name} complete!\n")

    print("=" * 80)
    print("Download Complete!")
    print("=" * 80)
    print(f"\nDataset structure:")
    print(f"{base_dir}/")
    print("  ├── train2014/")
    print("  ├── val2014/")
    print("  ├── test2014/")
    print("  └── annotations/")
    print()
    print("You can now use these datasets in your experiments.")
    print()

if __name__ == "__main__":
    main()
