# DEFT

DEFT is a tool for enzyme classification using protein language models and structural similarity search.

> **No GCP required.** AlphaFold structures are pulled over plain HTTPS from
> the public AlphaFold bucket and the EBI AlphaFold endpoint. Earlier
> versions of DEFT required a GCP service-account key and `gsutil`; that's
> no longer the case.

## Installation

### Install from source
```bash
git clone <repository-url>
cd DEFT
micromamba create -n deft python==3.10
micromamba activate deft
pip install -r requirements.txt
micromamba install -c conda-forge -c bioconda foldseek
pip install -e .
```

## Quick Start

The fastest path from a clean clone to predictions:

```bash
# Optional — set the cache directory (defaults to ~/.deft_cache)
export DEFT_CACHE=~/.deft_cache

# Download model weights and the training database from Zenodo
python deft.py download

# Run end-to-end prediction for a species (NCBI taxonomy ID)
python deft.py easy-predict --species-id 208964 --output-dir ./results
```

`easy-predict` will:
1. Download and cache the required model files (if not already present).
2. Pull the species' AlphaFold proteome over HTTPS.
3. Run prediction and save results to `./results/`.

## Cache Management

Model files are cached in `~/.deft_cache` by default. Override with `DEFT_CACHE`:

```bash
export DEFT_CACHE=/path/to/your/cache
```

Clear the cache with `rm -rf ~/.deft_cache`.

## Available Commands

| Command                  | What it does                                              |
| ------------------------ | --------------------------------------------------------- |
| `download`               | Download DEFT model weights and data from Zenodo.         |
| `easy-predict`           | Download (if needed) + create dataset + predict in one go.|
| `predict`                | Run prediction on a prepared dataset and alignment.       |
| `create-dataset`         | Build a dataset from a list of UniProt IDs.               |
| `create-dataset-species` | Build a dataset for an NCBI taxonomy ID.                  |
| `train`                  | Fine-tune a model with PEFT/LoRA.                         |
| `evaluate`               | Score predictions against labelled data.                  |
| `search`                 | Search a structural database for matches to a query.      |
| `annotate`               | Annotate a query CIF with EC predictions.                 |

Run `python deft.py <command> --help` for full options on any command.

## Examples

The examples below assume you've run `python deft.py download` first and
that `$DEFT` points at the extracted bundle:

```bash
export DEFT=${DEFT_CACHE:-~/.deft_cache}/DEFT
```

### `download` — fetch model weights and data
```bash
python deft.py download
# or force a fresh download:
python deft.py download --force
```

### `easy-predict` — one-shot species prediction
```bash
python deft.py easy-predict \
    --species-id 208964 \
    --output-dir ./results
```

### `create-dataset` — build a dataset from a UniProt ID list
```bash
# uniprot_list.txt: one accession per line (P00698, P00734, ...)
python deft.py create-dataset \
    --file     uniprot_list.txt \
    --output   ./data/manual \
    --mode     test \
    --train-db $DEFT/deft_aln/clean70_db
# Produces: ./data/manual/uniprot_list_res.csv  (ID, Sequence, 3DI)
#           ./data/manual/uniprot_list_aln.m8   (foldseek alignment)
```

### `create-dataset-species` — build a dataset for one taxonomy ID
```bash
python deft.py create-dataset-species \
    --species   208964 \
    --output    ./data/208964 \
    --train-db  $DEFT/deft_aln/clean70_db
```

### `predict` — run the model on a prepared dataset
```bash
python deft.py predict \
    --data      ./data/manual/uniprot_list_res.csv \
    --align     ./data/manual/uniprot_list_aln.m8 \
    --peft      $DEFT/deft_weights \
    --train-csv $DEFT/base.csv \
    --out       ./predictions.csv
```

### `annotate` — predict EC for a single CIF (or directory of CIFs)
```bash
python deft.py annotate \
    --query ./my_structures/AF-P00698-F1-model_v6.cif \
    --peft  $DEFT/deft_weights \
    --out   ./annotations.csv
# Without --out, predictions print to stdout.
```

### `search` — find structural neighbours and filter by predicted EC
```bash
python deft.py search \
    --query ./my_structures/AF-P0A6T1-F1-model_v6.cif \
    --db    $DEFT/deft_aln/clean70_db \
    --peft  $DEFT/deft_weights \
    --out   ./hits.csv
```

### `evaluate` — score predictions against labelled data
The input CSV must have an `EC` column.
```bash
python deft.py evaluate \
    --data      ./labelled_test.csv \
    --align     ./labelled_test_aln.m8 \
    --peft      $DEFT/deft_weights \
    --train-csv $DEFT/base.csv \
    --out       ./evaluation_results.csv
```

### `train` — fine-tune a new PEFT adapter
Heavy: needs a GPU and a labelled training CSV (`ID, Sequence, 3DI, EC`).
```bash
python deft.py train \
    --data      ./train.csv \
    --data-eval ./eval.csv \
    --save-path ./models/new_adapter
```

## File Formats

### Input CSV
- `ID` — Protein identifier
- `Sequence` — Amino-acid sequence
- `3DI` — 3Di structural sequence (optional for prediction; auto-derived from CIFs)

### Prediction output
- `Query` — Query protein ID
- `Target` — Target protein ID
- `Bits` — Foldseek alignment score
- `EC` — Predicted EC number

## Troubleshooting

### Mixed package versions on shared/HPC machines

If `import transformers` (or another dep) loads from `~/.local/lib/...`
instead of the env, an older user-site install is shadowing the env. Set
`PYTHONNOUSERSITE=1` for both `pip install` and runtime:

```bash
export PYTHONNOUSERSITE=1
```

### Missing model files
1. Re-run `python deft.py download --force`.
2. Confirm the URL in `cli/config.py` is reachable.
3. Set `DEFT_CACHE` to point at an existing extracted `DEFT/` directory.

### AlphaFold download failures
The species path lists tarballs via the public GCS JSON API and downloads
them over HTTPS. If listing fails, double-check the taxonomy ID at
[alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk/). The UniProt-list path
fetches each accession from EBI; missing IDs are reported and skipped.

## License

MIT — see the LICENSE file.

## Funding & Acknowledgements

This work was supported in part by a grant from the Army Research Office
(ARO 80093-CH-MUR) (to K.L. and D.K.) and the Karol Family Professorship
(to K.L.). We thank the Ribbeck lab for the purified mucin.
