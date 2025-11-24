import os
import sys
import tempfile
import urllib.request
import zipfile
import tarfile
import hashlib
import shutil
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
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models") + "/"
        self.eve_data_path = os.path.join(self.cache_dir, "EVE_DATA")
        self.peft_path = os.path.join(self.eve_data_path, "clean70_0_model")
        self.train_db_path = os.path.join(self.eve_data_path, "eve_aln", "clean70_db")
        self.train_csv_path = os.path.join(self.eve_data_path, "clean70_0_res.csv")

    def download_eve_data(self) -> bool:
        """Download and extract EVE_DATA.zip from Zenodo"""
        file_info = MODEL_URLS['eve_data']
        url = file_info['url']
        filename = file_info['filename']
        local_path = os.path.join(self.cache_dir, filename)
        
        try:
            print(f"Downloading {file_info['description']} from {url}...")
            
            # Create directory if it doesn't exist
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Download file with progress tracking
            def show_progress(block_num, block_size, total_size):
                if total_size > 0:
                    percent = min(100, (block_num * block_size * 100) // total_size)
                    print(f"\rDownload progress: {percent}%", end='', flush=True)
            
            urllib.request.urlretrieve(url, local_path, show_progress)
            print()  # New line after progress
            
            # Extract the zip file
            print(f"Extracting {local_path}...")
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir)
            
            # Remove the downloaded archive
            os.remove(local_path)
            
            print(f"Successfully downloaded and extracted {file_info['description']}")
            return True
            
        except Exception as e:
            print(f"Error downloading {file_info['description']}: {e}")
            return False

    def ensure_files_exist(self) -> bool:
        """Ensure all required files exist in cache, download if necessary"""
        # Check if EVE_DATA directory exists
        if not os.path.exists(self.eve_data_path):
            print(f"EVE_DATA directory not found at {self.eve_data_path}")
            print("Downloading EVE_DATA.zip from Zenodo...")
            if not self.download_eve_data():
                return False
        
        # Check if all required files exist
        files_to_check = [
            (self.peft_path, 'PEFT model'),
            (self.train_db_path, 'Training database'),
            (self.train_csv_path, 'Training CSV')
        ]
        
        all_files_exist = True
        
        for file_path, description in files_to_check:
            if not os.path.exists(file_path):
                print(f"Required file {description} not found at {file_path}")
                all_files_exist = False
        
        if not all_files_exist:
            print("\nSome required files are missing from the EVE_DATA directory.")
            print("Expected structure:")
            print(f"  {self.eve_data_path}/")
            print(f"    ├── clean70_0_model/ (PEFT model)")
            print(f"    ├── eve_aln/clean70_db/ (Training database)")
            print(f"    └── clean70_0_res.csv (Training CSV)")
            print("\nPlease check the downloaded EVE_DATA.zip contents.")
        
        return all_files_exist

    def run(self):
        """Run the easy-predict workflow"""
        print(f"EVE Easy Predict - Species ID: {self.species_id}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Output directory: {self.output_dir}")
        
        # Ensure all required files exist
        if not self.ensure_files_exist():
            print("Missing required files. Please check the EVE_DATA.zip download.")
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
