import os
import urllib.request
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from .predict import Predict
from .config import MODEL_URLS, DEFAULT_CACHE_DIR


@dataclass
class EasyPredict:
    species_id: int
    output_dir: str
    cache_dir: Optional[str] = None

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = os.environ.get(
                "DEFT_CACHE", os.path.expanduser(DEFAULT_CACHE_DIR)
            )

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.model_path = "westlake-repl/SaProt_650M_AF2"
        self.deft_path = os.path.join(self.cache_dir, "DEFT")
        self.peft_path = os.path.join(self.deft_path, "deft_weights")
        self.train_db_path = os.path.join(self.deft_path, "deft_aln", "clean70_db")
        self.train_csv_path = os.path.join(self.deft_path, "base.csv")

    def download_deft(self) -> bool:
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

    def ensure_files_exist(self) -> bool:
        """Ensure all required files exist in cache, download if necessary"""
        if not os.path.exists(self.deft_path):
            print(f"DEFT directory not found at {self.deft_path}")
            print("Downloading DEFT.zip from Zenodo...")
            if not self.download_deft():
                return False

        files_to_check = [
            (self.peft_path, "PEFT model"),
            (self.train_db_path, "Training database"),
            (self.train_csv_path, "Training CSV"),
        ]

        all_files_exist = True

        for file_path, description in files_to_check:
            if not os.path.exists(file_path):
                print(f"Required file {description} not found at {file_path}")
                all_files_exist = False

        if not all_files_exist:
            print("\nSome required files are missing from the DEFT directory.")
            print("Expected structure:")
            print(f"  {self.deft_path}/")
            print(f"    ├── deft_weights/ (PEFT model)")
            print(f"    ├── deft_aln/ (Training database)")
            print(f"    └── base.csv (Training CSV)")
            print("\nPlease check the downloaded DEFT.zip contents.")

        return all_files_exist

    def run(self):
        """Run the easy-predict workflow"""
        print(f"DEFT Easy Predict - Species ID: {self.species_id}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Output directory: {self.output_dir}")

        if not self.ensure_files_exist():
            print("Missing required files. Please check the DEFT.zip download.")
            return None

        from .create_dataset import CreateDatasetSpecies

        print(f"\nStep 1: Creating dataset for species {self.species_id}...")
        create_dataset = CreateDatasetSpecies(
            self.species_id, self.output_dir, self.train_db_path
        )
        create_dataset.run()

        print("\nStep 2: Running prediction...")
        data_file = os.path.join(self.output_dir, f"{self.species_id}.csv")
        align_file = os.path.join(self.output_dir, f"{self.species_id}_aln.m8")
        output_file = os.path.join(self.output_dir, f"{self.species_id}_pred.csv")

        if not os.path.exists(data_file):
            print(f"Error: Data file {data_file} not found")
            return None

        if not os.path.exists(align_file):
            print(f"Error: Alignment file {align_file} not found")
            return None

        predict_args = Predict(
            data=data_file,
            align=align_file,
            model=self.model_path,
            peft=self.peft_path,
            train_csv=self.train_csv_path,
            outfile=output_file,
        )

        result = predict_args.run()
        print(f"Prediction completed. Results saved to {output_file}")

        return result
