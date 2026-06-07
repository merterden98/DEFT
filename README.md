# DEFT

DEFT is a tool for enzyme classification using protein language models and structural similarity search.

## Installation

### Install from source
```bash
git clone <repository-url>
cd DEFT
micromamba create -n deft python==3.10 # users can also use conda/mamba/minimamba instead of micromamba interchangeably.
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
deft download

# Run end-to-end prediction for a species (NCBI taxonomy ID)
deft easy-predict --species-id 208964 --output-dir ./results
```

`easy-predict` will:
1. Download and cache the required model files (if not already present).
2. Pull the species' AlphaFold proteome over HTTPS.
3. Run prediction and save results to `./results/`.

> Invoke DEFT as `deft <command>` (installed by `pip install -e .`) or, equivalently, `python -m deft <command>`.

## Verify your installation

A self-contained test — **no input files required**. It downloads the
model bundle, fetches one real enzyme structure from AlphaFold, and predicts
its EC class.

```bash
# 1. Download model weights + data from Zenodo (~490 MB, one time)
deft download
export DEFT=${DEFT_CACHE:-~/.deft_cache}/DEFT

# 2. Fetch a known enzyme — human carbonic anhydrase 2 (true EC 4.2.1.1)
curl -L https://alphafold.ebi.ac.uk/files/AF-P00918-F1-model_v6.cif -o P00918.cif

# 3. Predict its EC class
deft annotate --query P00918.cif --peft $DEFT/deft_weights
```

DEFT predicts to the first two EC levels, so the expected output is:

```
P00918	4.2
```

### Check a panel of enzymes across EC classes

Drop several structures in one directory and annotate them in a single call
(one model load, all structures predicted together):

```bash
mkdir -p enzymes
for acc in P04040 P00558 P00760 P00918; do
    url=$(curl -s https://alphafold.ebi.ac.uk/api/prediction/$acc \
          | python -c "import sys,json; print(json.load(sys.stdin)[0]['cifUrl'])")
    curl -sL "$url" -o enzymes/$acc.cif
done

deft annotate --query enzymes --peft $DEFT/deft_weights --out preds.csv
```

Each predicted EC level-2 should match the enzyme's true class:

| Structure | Enzyme                      | True EC   | DEFT |
| --------- | --------------------------- | --------- | ---- |
| P04040    | Catalase                    | 1.11.1.6  | 1.11 |
| P00558    | Phosphoglycerate kinase 1   | 2.7.2.3   | 2.7  |
| P00760    | Trypsin                     | 3.4.21.4  | 3.4  |
| P00918    | Carbonic anhydrase 2        | 4.2.1.1   | 4.2  |

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

Run `deft <command> --help` for full options on any command.

## Examples

The examples below assume you've run `deft download` first and
that `$DEFT` points at the extracted bundle:

```bash
export DEFT=${DEFT_CACHE:-~/.deft_cache}/DEFT
```

### `download` — fetch model weights and data
```bash
deft download
# or force a fresh download:
deft download --force
```

### `easy-predict` — one-shot species prediction
```bash
deft easy-predict \
    --species-id 208964 \
    --output-dir ./results
```

### `create-dataset` — build a dataset from a UniProt ID list
```bash
# uniprot_list.txt: one accession per line (P00698, P00734, ...)
deft create-dataset \
    --file     uniprot_list.txt \
    --output   ./data/manual \
    --mode     test \
    --train-db $DEFT/deft_aln/clean70_db
# Produces: ./data/manual/uniprot_list_res.csv  (ID, Sequence, 3DI)
#           ./data/manual/uniprot_list_aln.m8   (foldseek alignment)
```

### `create-dataset-species` — build a dataset for one taxonomy ID
```bash
deft create-dataset-species \
    --species   208964 \
    --output    ./data/208964 \
    --train-db  $DEFT/deft_aln/clean70_db
```

### `predict` — run the model on a prepared dataset
```bash
deft predict \
    --data      ./data/manual/uniprot_list_res.csv \
    --align     ./data/manual/uniprot_list_aln.m8 \
    --peft      $DEFT/deft_weights \
    --train-csv $DEFT/base.csv \
    --out       ./predictions.csv
```

### `annotate` — predict EC for a single CIF (or directory of CIFs)
```bash
deft annotate \
    --query ./my_structures/AF-P00698-F1-model_v6.cif \
    --peft  $DEFT/deft_weights \
    --out   ./annotations.csv
# Without --out, predictions print to stdout.
```

### `search` — find structural neighbours and filter by predicted EC
```bash
deft search \
    --query ./my_structures/AF-P0A6T1-F1-model_v6.cif \
    --db    $DEFT/deft_aln/clean70_db \
    --peft  $DEFT/deft_weights \
    --out   ./hits.csv
```

### `evaluate` — score predictions against labelled data
The input CSV must have an `EC` column.
```bash
deft evaluate \
    --data      ./labelled_test.csv \
    --align     ./labelled_test_aln.m8 \
    --peft      $DEFT/deft_weights \
    --train-csv $DEFT/base.csv \
    --out       ./evaluation_results.csv
```

### `train` — fine-tune a new PEFT adapter
Heavy: needs a GPU and a labelled training CSV (`ID, Sequence, 3DI, EC`).
```bash
deft train \
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
1. Re-run `deft download --force`.
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
