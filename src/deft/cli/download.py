import os
import urllib.request
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from .config import MODEL_URLS, DEFAULT_CACHE_DIR


@dataclass
class Download:
    cache_dir: Optional[str] = None
    force: bool = False

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = os.environ.get(
                "DEFT_CACHE", os.path.expanduser(DEFAULT_CACHE_DIR)
            )

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.deft_path = os.path.join(self.cache_dir, "DEFT")
        self.peft_path = os.path.join(self.deft_path, "deft_weights")
        self.train_db_path = os.path.join(self.deft_path, "deft_aln", "clean70_db")
        self.train_csv_path = os.path.join(self.deft_path, "base.csv")

    def download_deft(self) -> bool:
        """Download and extract DEFT.zip from Zenodo"""
        file_info = MODEL_URLS["deft"]
        url = file_info["url"]
        filename = file_info["filename"]
        local_path = os.path.join(self.cache_dir, filename)

        try:
            print(f"Downloading {file_info['description']} from {url}...")

            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(f"\rDownload progress: {percent}%", end="", flush=True)

            urllib.request.urlretrieve(url, local_path, show_progress)
            print()

            print(f"Extracting {local_path}...")
            with zipfile.ZipFile(local_path, "r") as zip_ref:
                zip_ref.extractall(self.cache_dir)

            os.remove(local_path)

            print(f"Successfully downloaded and extracted {file_info['description']}")
            return True

        except Exception as e:
            print(f"Error downloading {file_info['description']}: {e}")
            return False

    def verify_files(self) -> bool:
        """Verify all required files exist after download"""
        files_to_check = [
            (self.peft_path, "PEFT model weights"),
            (self.train_db_path, "Training database"),
            (self.train_csv_path, "Training CSV"),
        ]

        all_files_exist = True

        for file_path, description in files_to_check:
            if os.path.exists(file_path):
                print(f"  ✓ {description}: {file_path}")
            else:
                print(f"  ✗ {description}: {file_path} (missing)")
                all_files_exist = False

        return all_files_exist

    def run(self) -> bool:
        """Run the download workflow"""
        print("DEFT Download")
        print(f"Cache directory: {self.cache_dir}")
        print()

        # Check if already downloaded
        if os.path.exists(self.deft_path) and not self.force:
            print(f"DEFT directory already exists at {self.deft_path}")
            print("\nVerifying files...")
            if self.verify_files():
                print("\nAll required files are present.")
                print("Use --force to re-download.")
                return True
            else:
                print("\nSome files are missing. Re-downloading...")

        # Download DEFT
        if not self.download_deft():
            print("Download failed.")
            return False

        # Verify files after download
        print("\nVerifying downloaded files...")
        if self.verify_files():
            print("\nDownload completed successfully!")
            print(f"\nDEFT files are available at: {self.deft_path}")
            print("\nYou can now use these paths with other DEFT commands:")
            print(f"  --peft {self.peft_path}")
            print(f"  --train-db {self.train_db_path}")
            print(f"  --train-csv {self.train_csv_path}")
            return True
        else:
            print("\nDownload completed but some files are missing.")
            print("Please check the downloaded DEFT.zip contents.")
            return False
