import os
import sys
import tempfile
import urllib.request
import zipfile
import tarfile
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from .predict import Predict
from .config import MODEL_URLS, FILE_CHECKSUMS, DEFAULT_CACHE_DIR


@dataclass
class EasyPredict:
    species_id: int
    output_dir: str
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        # Set up cache directory
        if self.cache_dir is None:
            self.cache_dir = os.environ.get('EVE_CACHE', os.path.expanduser(DEFAULT_CACHE_DIR))
        
        # Create cache directory if it doesn't exist
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Define file paths in cache
        self.model_path = os.path.join(os.getcwd(), "models")
        self.peft_path = os.path.join(self.cache_dir, "peft_model")
        self.train_db_path = os.path.join(self.cache_dir, "train_db")
        self.train_csv_path = os.path.join(self.cache_dir, "train_csv.csv")

    def download_file(self, file_type: str, local_path: str) -> bool:
        """Download a file from URL to local path"""
        if file_type not in MODEL_URLS:
            print(f"Unknown file type: {file_type}")
            return False
            
        file_info = MODEL_URLS[file_type]
        url = file_info['url']
        extract = file_info['extract']
        
        try:
            print(f"Downloading {file_info['description']} from {url}...")
            
            # Create directory if it doesn't exist
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Download file with progress tracking
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(f"\rDownload progress: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, local_path, show_progress)
            print()  # New line after progress
            
            # Extract if needed
            if extract:
                print(f"Extracting {local_path}...")
                if local_path.endswith('.zip'):
                    with zipfile.ZipFile(local_path, 'r') as zip_ref:
                        zip_ref.extractall(Path(local_path).parent)
                elif local_path.endswith('.tar.gz'):
                    with tarfile.open(local_path, 'r:gz') as tar_ref:
                        tar_ref.extractall(Path(local_path).parent)
                
                # Remove the downloaded archive
                os.remove(local_path)
            
            print(f"Successfully downloaded and processed {file_info['description']}")
            return True
            
        except Exception as e:
            print(f"Error downloading {file_info['description']}: {e}")
            return False

    def ensure_files_exist(self) -> bool:
        """Ensure all required files exist in cache, download if necessary"""
        files_to_check = [
            (self.model_path, 'model'),
            (self.peft_path, 'peft'),
            (self.train_db_path, 'train_db'),
            (self.train_csv_path, 'train_csv')
        ]
        
        all_files_exist = True
        
        for file_path, file_type in files_to_check:
            if not os.path.exists(file_path):
                print(f"Required file {file_path} not found in cache.")
                all_files_exist = False
                
                # Try to download the file
                if file_type in MODEL_URLS:
                    print(f"Attempting to download {file_type}...")
                    if self.download_file(file_type, file_path):
                        all_files_exist = True
                    else:
                        all_files_exist = False
                else:
                    print(f"No download URL configured for {file_type}")
                    all_files_exist = False
        
        if not all_files_exist:
            print("\nSome required files could not be downloaded.")
            print("Please manually download the files to:")
            print(f"  Model: {self.model_path}")
            print(f"  PEFT: {self.peft_path}")
            print(f"  Train DB: {self.train_db_path}")
            print(f"  Train CSV: {self.train_csv_path}")
            print("\nOr update the URLs in cli/config.py")
        
        return all_files_exist

    def run(self):
        """Run the easy-predict workflow"""
        print(f"EVE Easy Predict - Species ID: {self.species_id}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Ensure all required files exist
        if not self.ensure_files_exist():
            print("Missing required files. Please download them manually or update the URLs.")
            return None
        
        # Import here to avoid circular imports
        from .create_dataset import CreateDatasetSpecies
        
        # Step 1: Create dataset for the species
        print(f"\nStep 1: Creating dataset for species {self.species_id}...")
        create_dataset = CreateDatasetSpecies(
            self.species_id, 
            self.output_dir, 
            self.train_db_path
        )
        create_dataset.run()
        
        # Step 2: Run prediction
        print(f"\nStep 2: Running prediction...")
        data_file = os.path.join(self.output_dir, f"{self.species_id}_res.csv")
        align_file = os.path.join(self.output_dir, f"{self.species_id}_aln.m8")
        output_file = os.path.join(self.output_dir, f"{self.species_id}_pred.csv")
        
        # Check if files exist
        if not os.path.exists(data_file):
            print(f"Error: Data file {data_file} not found")
            return None
        
        if not os.path.exists(align_file):
            print(f"Error: Alignment file {align_file} not found")
            return None
        
        # Run prediction
        predict_args = Predict(
            data=data_file,
            align=align_file,
            model=self.model_path,
            peft=self.peft_path,
            train_csv=self.train_csv_path,
            outfile=output_file
        )
        
        result = predict_args.run()
        print(f"Prediction completed. Results saved to {output_file}")
        
        return result
