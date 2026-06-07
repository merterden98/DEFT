# DEFT CLI API

Reference for every DEFT command, its options, inputs, and outputs.

DEFT classifies enzymes by combining a protein language model (SaProt-650M)
with foldseek structural similarity search. Most commands predict an **EC
number** to its first two levels (e.g. `3.4`) per protein.

## Invocation

DEFT is exposed as a console script and as a module — these are equivalent:

```bash
deft <command> [options]
python -m deft <command> [options]
```

Run `deft <command> --help` for the authoritative, always-current option list.

## Conventions

- **Required** options are marked ✷; all others are optional with the default
  shown.
- Paths may be relative or absolute.
- Structure inputs are mmCIF (`.cif` / `.cif.gz`). A `--query` that is a
  **directory** is processed as a batch (all CIFs in it).
- On any foldseek/subprocess failure DEFT **fails fast** with a `RuntimeError`
  carrying the command and its stderr, rather than continuing silently.

## Environment variables

| Variable          | Used by                          | Meaning                                                              |
| ----------------- | -------------------------------- | ------------------------------------------------------------------- |
| `DEFT_CACHE`      | `download`, `easy-predict`       | Cache directory for model weights + data. Default `~/.deft_cache`.  |
| `HF_HOME`         | all model-loading commands       | HuggingFace cache for the SaProt-650M base model (auto-downloaded). |
| `PYTHONNOUSERSITE`| all (on shared/HPC machines)     | Set to `1` so `~/.local` packages don't shadow the env.             |

## Cache layout

`download` (and `easy-predict`) extract the Zenodo bundle into `$DEFT_CACHE/DEFT/`:

```
$DEFT_CACHE/DEFT/
├── deft_weights/            # PEFT (LoRA) adapter + tokenizer   → --peft
├── deft_aln/clean70_db      # foldseek training database         → --db / --train-db
└── base.csv                 # training table (ID,Sequence,3DI,EC) → --train-csv
```

A convenience shorthand used throughout the examples:

```bash
export DEFT=${DEFT_CACHE:-~/.deft_cache}/DEFT
```

---

## Commands

### `download`

Download and extract the DEFT model weights + data bundle (~490 MB) from Zenodo.

| Option        | Type | Default                       | Description                                  |
| ------------- | ---- | ----------------------------- | -------------------------------------------- |
| `--cache-dir` | str  | `$DEFT_CACHE` or `~/.deft_cache` | Where to extract the bundle.              |
| `--force`     | flag | `false`                       | Re-download even if the bundle exists.       |

**Outputs:** the cache layout above. Idempotent — skips download if present
(unless `--force`).

```bash
deft download
```

---

### `easy-predict`

One-shot pipeline for a whole species: download bundle (if needed) → fetch the
species' AlphaFold structures → build dataset → predict EC numbers.

| Option         | Type | Default        | Description                                                    |
| -------------- | ---- | -------------- | -------------------------------------------------------------- |
| `--species-id` ✷ | int  | —              | NCBI taxonomy ID.                                              |
| `--output-dir` ✷ | str  | —              | Directory for intermediate and output files.                  |
| `--cache-dir`  | str  | `$DEFT_CACHE`  | Override the model/data cache directory.                       |

**Outputs** (in `--output-dir`):

| File                     | Contents                                            |
| ------------------------ | --------------------------------------------------- |
| `<species-id>_res.csv`   | Dataset: `ID,Sequence,3DI`                          |
| `<species-id>_aln.m8`    | foldseek alignment to the training database         |
| `<species-id>_pred.csv`  | Final predictions (alignment rows + EC columns)     |

```bash
deft easy-predict --species-id 1679 --output-dir ./results
```

> The species path uses the curated AlphaFold proteome tarball when one exists,
> otherwise it resolves the UniProt reference proteome and fetches each CIF
> individually from EBI. Missing accessions are reported and skipped.

---

### `create-dataset`

Build a dataset (and optionally an alignment) from an explicit list of UniProt
accessions.

| Option        | Type | Default   | Description                                                          |
| ------------- | ---- | --------- | ------------------------------------------------------------------- |
| `--file` ✷    | str  | —         | Text file of UniProt accessions, one per line.                      |
| `--output` ✷  | str  | —         | Output directory.                                                   |
| `--ec`        | str  | `None`    | Tab-separated `UniProtID<TAB>EC` annotation file.                   |
| `--test-file` | str  | `None`    | UniProt accessions for a test set (with `--mode train`).            |
| `--test-ec`   | str  | `None`    | EC annotation file for the test set.                                |
| `--train-db`  | str  | `None`    | foldseek training DB. **Required when `--mode test`.**              |
| `--mode`      | str  | `train`   | `train` or `test` (case-insensitive).                              |

**Outputs** (`<name>` = the `--file` basename without extension):

| File                  | Contents                                                  |
| --------------------- | -------------------------------------------------------- |
| `<name>_res.csv`      | `ID,Sequence,3DI` (plus `EC` when `--ec` is given)       |
| `<name>_aln.m8`       | foldseek alignment (when `--mode test`, or with a test set) |
| `<name>_res_test.csv` | test-set dataset (when `--test-file`/`--test-ec` given)  |

```bash
# uniprot_list.txt: one accession per line
deft create-dataset \
    --file     uniprot_list.txt \
    --output   ./data/manual \
    --mode     test \
    --train-db $DEFT/deft_aln/clean70_db
```

---

### `create-dataset-species`

Build a dataset + alignment for one species from its AlphaFold structures.
(Lower-level than `easy-predict`: it stops after the dataset/alignment, without
running prediction.)

| Option        | Type | Default | Description                          |
| ------------- | ---- | ------- | ------------------------------------ |
| `--species` ✷ | str  | —       | Species identifier (NCBI taxonomy ID). |
| `--output` ✷  | str  | —       | Output directory.                    |
| `--train-db` ✷ | str  | —       | foldseek training database path.     |

**Outputs:** `<species>_res.csv` (`ID,Sequence,3DI`) and `<species>_aln.m8`.

```bash
deft create-dataset-species \
    --species  1679 \
    --output   ./data/1679 \
    --train-db $DEFT/deft_aln/clean70_db
```

---

### `predict`

Run EC prediction on a prepared dataset + alignment.

| Option        | Type | Default | Description                                       |
| ------------- | ---- | ------- | ------------------------------------------------- |
| `--data` ✷    | str  | —       | Dataset CSV (`ID,Sequence,3DI`).                  |
| `--align` ✷   | str  | —       | foldseek alignment `.m8` file.                    |
| `--peft` ✷    | str  | —       | PEFT adapter weights directory.                   |
| `--train-csv` ✷ (alias `--train_csv`) | str | — | Training CSV used to label alignment targets.     |
| `--outfile` ✷ (alias `--out`)          | str | — | Output CSV path.                                  |

**Output:** alignment rows whose predicted EC (2-level) matches the target's,
with columns `Query,Target,…,Bits,EC,EC_2,Query_EC2`.

```bash
deft predict \
    --data      ./data/manual/uniprot_list_res.csv \
    --align     ./data/manual/uniprot_list_aln.m8 \
    --peft      $DEFT/deft_weights \
    --train-csv $DEFT/base.csv \
    --out       ./predictions.csv
```

---

### `annotate`

Predict EC numbers for a single CIF (or a directory of CIFs) — no alignment
required. The lightest way to classify structures.

| Option      | Type | Default | Description                                        |
| ----------- | ---- | ------- | -------------------------------------------------- |
| `--query` ✷ | str  | —       | Query CIF file, or a directory of CIFs.            |
| `--peft` ✷  | str  | —       | PEFT adapter weights directory.                    |
| `--out`     | str  | `None`  | Output CSV (`ID,EC`). If unset, prints to stdout.  |

**Output:** `ID<TAB>EC` to stdout, and `ID,EC` CSV when `--out` is given.

```bash
deft annotate --query AF-P00918-F1-model_v6.cif --peft $DEFT/deft_weights
# P00918	4.2
```

---

### `search`

Find structural neighbours of a query in a foldseek database, keeping only hits
whose predicted EC agrees between query and target.

| Option      | Type | Default | Description                       |
| ----------- | ---- | ------- | --------------------------------- |
| `--query` ✷ | str  | —       | Query CIF file.                   |
| `--db` ✷    | str  | —       | foldseek database to search.      |
| `--peft` ✷  | str  | —       | PEFT adapter weights directory.   |
| `--out` ✷   | str  | —       | Output CSV of filtered alignments. |

```bash
deft search \
    --query AF-P0A6T1-F1-model_v6.cif \
    --db    $DEFT/deft_aln/clean70_db \
    --peft  $DEFT/deft_weights \
    --out   ./hits.csv
```

---

### `evaluate`

Score predictions against labelled data (input CSV must include an `EC` column).

| Option        | Type | Default | Description                                                       |
| ------------- | ---- | ------- | ---------------------------------------------------------------- |
| `--data` ✷    | str  | —       | Labelled dataset CSV (`ID,Sequence,3DI,EC`).                     |
| `--align` ✷   | str  | —       | foldseek alignment `.m8` file.                                   |
| `--train-csv` ✷ (alias `--train_csv`) | str | — | Training CSV used to label alignment targets.                   |
| `--peft` ✷    | str  | —       | PEFT adapter weights directory.                                 |
| `--out` ✷     | str  | —       | Output CSV (alignments + EC labels).                            |
| `--skip-prediction-filter` | flag | `false` | Skip model prediction; only attach EC labels from training data. (`--no-skip-prediction-filter` to force off.) |

**Output:** alignment CSV with EC labels; precision/recall/F1/accuracy are
computed internally (CLEAN-style metrics).

```bash
deft evaluate \
    --data      ./labelled_test.csv \
    --align     ./labelled_test_aln.m8 \
    --peft      $DEFT/deft_weights \
    --train-csv $DEFT/base.csv \
    --out       ./evaluation_results.csv
```

---

### `train`

Fine-tune a new PEFT/LoRA adapter. **Heavy** — needs a GPU and labelled CSVs.

| Option        | Type  | Default            | Description                                                   |
| ------------- | ----- | ------------------ | ------------------------------------------------------------ |
| `--data` ✷    | str   | —                  | Training CSV (`ID,Sequence,3DI,EC`).                         |
| `--data-eval` ✷ | str | —                  | Evaluation CSV.                                              |
| `--save-path` ✷ | str | —                  | Directory for the trained adapter + stats.                  |
| `--model`     | str   | bundled `models/`  | Base model directory or HF identifier.                       |
| `--lr`        | float | `5e-5`             | Learning rate.                                               |
| `--epochs`    | int   | `10`               | Number of training epochs.                                  |
| `--ec-level`  | int   | `2`                | EC digits to train on: `2` (first two) or `4` (full EC).    |

**Outputs** (in `--save-path`): the adapter, `labels.json` (label map), and
`stats.json` (training results).

```bash
deft train \
    --data      ./train.csv \
    --data-eval ./eval.csv \
    --save-path ./models/new_adapter \
    --ec-level  2
```

---

## File formats

### Dataset CSV (`--data`)

| Column     | Required | Description                                         |
| ---------- | -------- | --------------------------------------------------- |
| `ID`       | yes      | Protein identifier (bare UniProt accession).        |
| `Sequence` | yes      | Amino-acid sequence.                                |
| `3DI`      | yes      | foldseek 3Di structural sequence.                   |
| `EC`       | training/eval | EC number (e.g. `3.4.21.4`); multiple separated by `;`. |

### Alignment file (`.m8`, `--align`)

Tab-separated foldseek output, 12 columns:
`Query, Target, Fident, Alnlen, Mismatch, Gapopen, Qstart, Qend, Tstart, Tend, Evalue, Bits`.
DEFT strips the `.cif`/EC suffix from `Query`/`Target` to match dataset `ID`s.

### Prediction output (`predict` / `easy-predict`)

The alignment table filtered/augmented with EC columns: `EC` (target's EC),
`EC_2` (its first two levels), and `Query_EC2` (the query's predicted EC, two
levels). Rows are kept where `EC_2 == Query_EC2` and capped at 1000 per query
by descending `Bits`.

### Search output (`search`)

The alignment table with `Query_Pred` (the query's predicted EC) and
`Target_Pred` (the target's predicted EC), filtered to rows where the two agree.

### Evaluation output (`evaluate`)

The alignment table with `EC`, `EC_2`, `Query_EC2`, and `True_EC` columns;
precision/recall/F1/accuracy (CLEAN-style) are computed internally.

### Annotation output (`annotate`)

`ID,EC` — one predicted EC (two levels) per structure.

---

## Programmatic API

The CLI is a thin shell over importable functions:

```python
from deft.utils.loader import retrieve_model, construct_query, predict_ec

tokenizer, model = retrieve_model("westlake-repl/SaProt_650M_AF2", peft_dir)
dataset = construct_query("structure.cif", tokenizer)
predictions = predict_ec(model, tokenizer, dataset)   # -> [(id, ec), ...]
```

Module layout:

| Module               | Responsibility                                                        |
| -------------------- | -------------------------------------------------------------------- |
| `deft.utils.loader`  | Model/tokenizer loading, dataset construction, `predict_ec` (the EC-prediction seam). |
| `deft.utils.foldseek`| The single adapter to the foldseek binary: `align`, `extract_3di_from_db`, `retrieve_3di`. |
| `deft.utils.alignment`| Alignment parsing + EC analysis: `read_aln`, `assign_predictions`, `restrict_aln`, `add_ec_data`. |
| `deft.utils.query_alphafold` | Fetch AlphaFold structures by accession or taxonomy ID.       |
| `deft.cli`           | Typer command definitions.                                           |

See [CONTEXT.md](CONTEXT.md) for domain vocabulary (EC number, EC prediction,
3Di, Alignment).
