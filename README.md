# DEFT

DEFT is a tool for enzyme classification using protein language models and structural similarity search.

## Installation

### Install from source
```bash
git clone <repository-url>
cd EVE
micromamba create -n deft python==3.10
micromamba activate deft
pip install -r requirements.txt
micromamba install -c conda-forge -c bioconda foldseek
pip install -e .
```



## GCP / AlphaFold Setup

Some commands (such as dataset creation from AlphaFold) download structures from Google Cloud.
To enable this, you must configure:

- A **GCP service account JSON key** with access to the AlphaFold public bucket and BigQuery.
- The `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to that key.
- The `gsutil` CLI tool (installed via the Google Cloud SDK).

### 1. Install Google Cloud SDK / gsutil

[Follow the official Google Cloud SDK installation instructions for your platform.](https://docs.cloud.google.com/sdk/docs/install-sdk)
After installation, verify that `gsutil` is available:

```bash
gsutil version
```

### 2. Create a service account and key

1. In the GCP Console, go to **IAM & Admin → Service Accounts**.
2. Create a service account (or reuse an existing one) and grant it at least
   storage read access to the AlphaFold bucket and BigQuery read access.
3. Create a **JSON key** for this service account and download it to a secure location.

### 3. Set GOOGLE_APPLICATION_CREDENTIALS

Point `GOOGLE_APPLICATION_CREDENTIALS` to your JSON key file, for example:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

### 4. Check your GCP setup

DEFT provides a small helper command to verify that both credentials and `gsutil`
are configured correctly:

```bash
python deft.py gcp-check
```

This command will:

- Confirm that `GOOGLE_APPLICATION_CREDENTIALS` is set and points to an existing file.
- Check that `gsutil` is available on your `PATH`.




## Quick Start

### Downloading DEFT Data

Before running predictions, you need to download the DEFT model weights and data. Use the `download` command:

```bash
# Set cache directory (optional, defaults to ~/.deft_cache)
export DEFT_CACHE=~/.deft_cache

# Download DEFT data
python deft.py download

# Force re-download if files already exist
python deft.py download --force

# Specify a custom cache directory
python deft.py download --cache-dir /path/to/cache
```

The `download` command will:
1. Download DEFT.zip from Zenodo
2. Extract to the cache directory
3. Verify all required files are present

### Easy-Predict

The `easy-predict` command provides a simplified workflow that automatically downloads files (if needed) and runs prediction:

```bash
# Run prediction for a species
python deft.py easy-predict --species-id 208964 --output-dir ./results
```

The `easy-predict` command will:
1. Download and cache the required model files (if not already present)
2. Create a dataset for the specified species
3. Run prediction and save results


## Cache Management

Model files are cached in `~/.deft_cache` by default. You can change this by setting the `DEFT_CACHE` environment variable:

```bash
export DEFT_CACHE=/path/to/your/cache
```

To clear the cache:
```bash
rm -rf ~/.deft_cache
```


## Available Commands

- `download`: Download DEFT model weights and data files
- `easy-predict`: Simplified prediction with automatic file management
- `predict`: Manual prediction with full control over paths
- `create-dataset`: Create dataset from UniProt IDs
- `create-dataset-species`: Create dataset from NCBI taxonomy ID
- `train`: Train a new model
- `evaluate`: Evaluate model performance
- `search`: Search for similar proteins
- `annotate`: Annotate proteins with EC numbers

## Examples

### Create dataset for a species
```bash
python deft.py create-dataset-species \
    --species 208964 \
    --output ./data/208964 \
    --train-db /path/to/train_db
```

### Train a model
```bash
python deft.py train \
    --data /path/to/train_data.csv \
    --save-path ./models/new_model
```

### Evaluate model
```bash
python deft.py evaluate \
    --data /path/to/test_data.csv \
    --align /path/to/alignment.m8 \
    --peft ./peft_model/ \
    --train-csv /path/to/train.csv \
    --out ./evaluation_results.csv
```

## File Formats

### Input CSV Format
The input CSV should contain the following columns:
- `ID`: Protein identifier
- `Sequence`: Amino acid sequence
- `3DI`: 3D interaction sequence (optional for prediction)

### Output Format
The prediction output contains:
- `Query`: Query protein ID
- `Target`: Target protein ID  
- `Bits`: Alignment score
- `EC`: Predicted EC number

## Troubleshooting

### Missing Model Files
If you get errors about missing model files:

1. Check that the URLs in `cli/config.py` are correct
2. Manually download files to the cache directory
3. Set the `DEFT_CACHE` environment variable to point to your files


## License

This project is licensed under the MIT License - see the LICENSE file for details.
