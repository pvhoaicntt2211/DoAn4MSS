#!/usr/bin/env python3
"""
Script to automatically download and extract MUSDB18-HQ dataset from Zenodo.
Organizes data into the expected structure for 4-stem separation.
"""

import argparse
import os
import sys
import shutil
import zipfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm


# MUSDB18-HQ Zenodo URLs
MUSDB18_HQ_URL = "https://zenodo.org/record/3338373/files/musdb18hq.zip"


class DownloadProgressBar(tqdm):
    """Progress bar for download with urllib"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urlretrieve(url, filename=output_path, reporthook=t.update_to)


def check_existing_data(output_dir):
    """Check if MUSDB18 data already exists"""
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    
    if train_dir.exists() and test_dir.exists():
        # Check if directories have content
        train_songs = [d for d in train_dir.iterdir() if d.is_dir()]
        test_songs = [d for d in test_dir.iterdir() if d.is_dir()]
        
        if len(train_songs) > 0 and len(test_songs) > 0:
            return True
    return False


def organize_musdb18_structure(musdb_root, output_dir):
    """
    Organize MUSDB18-HQ into the expected structure.
    MUSDB18-HQ comes with train/ and test/ folders, each containing song folders
    with individual stem files (vocals.wav, drums.wav, bass.wav, other.wav, mixture.wav)
    """
    print("\n📂 Organizing data structure...")
    
    musdb_path = Path(musdb_root)
    output_path = Path(output_dir)
    
    # The zip should contain a musdb18hq folder
    if not musdb_path.exists():
        print(f"❌ Source path not found: {musdb_path}")
        return False
    
    # Look for the actual data directory
    possible_paths = [
        musdb_path / "musdb18hq",
        musdb_path,
    ]
    
    data_root = None
    for p in possible_paths:
        if (p / "train").exists() or (p / "test").exists():
            data_root = p
            break
    
    if data_root is None:
        print(f"❌ Could not find train/test directories in extracted data")
        return False
    
    print(f"✓ Found MUSDB18-HQ data at: {data_root}")
    
    # Copy train and test directories
    for split in ['train', 'test']:
        src_split = data_root / split
        dst_split = output_path / split
        
        if not src_split.exists():
            print(f"⚠️ Warning: {split} directory not found, skipping")
            continue
        
        print(f"  Copying {split}/ ...")
        if dst_split.exists():
            shutil.rmtree(dst_split)
        
        dst_split.mkdir(parents=True, exist_ok=True)
        
        # Copy each song folder
        song_folders = [d for d in src_split.iterdir() if d.is_dir()]
        for song_folder in tqdm(song_folders, desc=f"  {split} songs"):
            dst_song = dst_split / song_folder.name
            shutil.copytree(song_folder, dst_song)
    
    print(f"✅ Data organized successfully at: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and setup MUSDB18-HQ dataset for 4-stem separation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for organized data (default: data/)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists"
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary download files after extraction"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Check if data already exists
    if not args.force and check_existing_data(output_dir):
        print(f"✓ MUSDB18 data already exists at: {output_dir}")
        print("  Use --force to re-download")
        
        # Show statistics
        train_songs = len([d for d in (output_dir / "train").iterdir() if d.is_dir()])
        test_songs = len([d for d in (output_dir / "test").iterdir() if d.is_dir()])
        print(f"\n📊 Dataset Statistics:")
        print(f"  Train songs: {train_songs}")
        print(f"  Test songs: {test_songs}")
        return 0
    
    print("🎵 MUSDB18-HQ Downloader")
    print("=" * 50)
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Download
        print("📥 Downloading MUSDB18-HQ from Zenodo...")
        print("   (This is ~30GB and may take a while...)")
        zip_path = temp_path / "musdb18hq.zip"
        
        try:
            download_url(MUSDB18_HQ_URL, zip_path)
            print(f"✓ Downloaded to: {zip_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return 1
        
        # Extract
        print("\n📦 Extracting archive...")
        extract_path = temp_path / "extracted"
        extract_path.mkdir(exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract with simple progress based on file count
                members = zip_ref.namelist()
                
                with tqdm(total=len(members), unit='files', desc="Extracting") as pbar:
                    for member in members:
                        zip_ref.extract(member, extract_path)
                        pbar.update(1)
            
            print(f"✓ Extracted to: {extract_path}")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return 1
        
        # Organize structure
        if not organize_musdb18_structure(extract_path, output_dir):
            print("❌ Failed to organize data structure")
            return 1
        
        if args.keep_temp:
            print(f"\n💾 Temporary files kept at: {temp_dir}")
        else:
            print("\n🗑️  Cleaning up temporary files...")
    
    # Show final statistics
    train_songs = len([d for d in (output_dir / "train").iterdir() if d.is_dir()])
    test_songs = len([d for d in (output_dir / "test").iterdir() if d.is_dir()])
    
    print("\n" + "=" * 50)
    print("✅ MUSDB18-HQ setup complete!")
    print(f"\n📊 Dataset Statistics:")
    print(f"  Train songs: {train_songs}")
    print(f"  Test songs: {test_songs}")
    print(f"\n📁 Data location: {output_dir.absolute()}")
    print("\nYou can now train the model with:")
    print(f"  python train.py --train-dir {output_dir}/train --valid-dir {output_dir}/test")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
