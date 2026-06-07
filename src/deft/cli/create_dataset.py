import typing as T
import pandas as pd
from dataclasses import dataclass
import sys
import os
from pathlib import Path

from ..utils import query_alphafold
from ..utils import foldseek


@dataclass
class CreateDataset:
    file: str
    output: str  # Output path
    ec: T.Optional[str]
    test_file: T.Optional[str]
    test_ec: T.Optional[str]
    train_db: T.Optional[str]
    mode: str = "train"

    def run(self):
        main(self)


@dataclass
class CreateDatasetSpecies:
    species: str
    output: str
    train_db: str

    def run(self):
        main_species(self)


def prepare_structures(folder, ec_data):
    """Clean an AlphaFold structure folder in place, then extract (aa, 3Di) records.

    AlphaFold proteome tarballs ship .pdb.gz, confidence JSONs, and PAE
    artifacts alongside the CIFs; foldseek picks up any parseable structure
    file, so leaving them behind doubles every entry. Drop everything that
    isn't a CIF and rename each CIF to `<uniprot>[_<ec>].<ext>`, then hand the
    folder to the foldseek adapter.

    Returns (aa_records, struct_records, db_path) — the db path is reusable as
    an alignment target.
    """
    for root, _, files in os.walk(folder):
        for fname in files:
            src = os.path.join(root, fname)
            if fname.endswith(".cif.gz"):
                ext = ".cif.gz"
            elif fname.endswith(".cif"):
                ext = ".cif"
            else:
                try:
                    os.remove(src)
                except OSError:
                    pass
                continue
            parts = fname.split("-")
            if len(parts) < 2:
                continue
            uniprot_id = parts[1]
            if ec_data is not None and uniprot_id in ec_data:
                new_name = f"{uniprot_id}_{ec_data[uniprot_id]}{ext}"
            else:
                new_name = f"{uniprot_id}{ext}"
            dst = os.path.join(root, new_name)
            if src != dst:
                try:
                    os.rename(src, dst)
                except OSError:
                    pass

    return foldseek.retrieve_3di(folder)


def main_species(args: CreateDatasetSpecies):
    import os

    tmp_folder = str(Path(args.output))

    # Check if args.output exists
    if not os.path.exists(args.output):
        print(f"Output folder {args.output} does not exist", file=sys.stderr)
        print(f"Creating output folder {args.output}", file=sys.stderr)

        # recursively create the output folder if it does not exist
        Path(args.output).mkdir(parents=True, exist_ok=True)

    name = f"{args.species}"
    tmp_folder_train = query_alphafold.get_pdb_files([], tmp_folder, name, query=name)
    print(tmp_folder_train)

    seq_records, seq_records_struct, traindb_path = prepare_structures(
        tmp_folder_train, None
    )

    # foldseek's fasta keys come from the CIF filenames, so they keep the ".cif"
    # (or ".cif.gz") suffix. read_aln strips that off the alignment Query/Target
    # columns, so we have to strip it here too — otherwise the EC-prefix join in
    # alignment.assign_predictions silently produces zero rows.
    items = []
    for raw_id in seq_records.keys():
        bare_id = raw_id.split(".")[0]
        items.append(
            {
                "ID": bare_id,
                "Sequence": str(seq_records[raw_id].seq),
                "3DI": str(seq_records_struct[raw_id].seq),
            }
        )

    df = pd.DataFrame(items)
    df.to_csv(f"{args.output}/{name}_res.csv", index=False)

    foldseek.align(tmp_folder_train, args.train_db, f"{args.output}/{name}_aln.m8")


def main(args: CreateDataset):
    with open(args.file, "r") as f:
        uniprot_ids = f.read().splitlines()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    tmp_folder = str(Path(args.output))

    ec_data = None
    if args.ec:
        with open(args.ec, "r") as f:
            ec_data = f.read().splitlines()
            ec_data = {line.split("\t")[0]: line.split("\t")[1] for line in ec_data}

        for uniprot_id in uniprot_ids:
            if uniprot_id not in ec_data:
                print(
                    f"Uniprot ID {uniprot_id} not found in the EC data", file=sys.stderr
                )
                sys.exit(1)

    # get filename of args.file
    name = os.path.basename(args.file)
    name = name.split(".")[0]
    tmp_folder_train = query_alphafold.get_pdb_files(uniprot_ids, tmp_folder, name)

    seq_records, seq_records_struct, traindb_path = prepare_structures(
        tmp_folder_train, ec_data
    )

    items = []
    missing = 0
    for uniprot_id in uniprot_ids:
        try:
            if args.ec and ec_data is not None:
                items.append(
                    {
                        "ID": uniprot_id,
                        "Sequence": str(seq_records[uniprot_id].seq),
                        "3DI": str(seq_records_struct[uniprot_id].seq),
                        "EC": ec_data[uniprot_id],
                    }
                )
            else:
                items.append(
                    {
                        "ID": uniprot_id,
                        "Sequence": str(seq_records[uniprot_id].seq),
                        "3DI": str(seq_records_struct[uniprot_id].seq),
                    }
                )
        except KeyError:
            missing += 1

    if missing > 0:
        print(f"Missing {missing} uniprot ids", file=sys.stderr)

    df = pd.DataFrame(items)
    df.to_csv(f"{args.output}/{name}_res.csv", index=False)

    if args.mode == "test":
        assert args.train_db is not None, "train_db is required when mode is test"
        foldseek.align(
            tmp_folder_train, args.train_db, f"{args.output}/{name}_aln.m8"
        )

    if args.test_ec and args.test_file and args.mode == "train":
        with open(args.test_file, "r") as f:
            uniprot_ids = f.read().splitlines()
        ec_data = None
        with open(args.test_ec, "r") as f:
            ec_data = f.read().splitlines()
            ec_data = {line.split("\t")[0]: line.split("\t")[1] for line in ec_data}

        tmp_folder_test = query_alphafold.get_pdb_files(
            uniprot_ids, tmp_folder, f"{name}_test"
        )
        seq_records, seq_records_struct, _ = prepare_structures(tmp_folder_test, ec_data)

        items = []
        missing = 0
        for uniprot_id in uniprot_ids:
            try:
                if args.test_ec and ec_data is not None:
                    items.append(
                        {
                            "ID": uniprot_id,
                            "Sequence": str(seq_records[uniprot_id].seq),
                            "3DI": str(seq_records_struct[uniprot_id].seq),
                            "EC": ec_data[uniprot_id],
                        }
                    )
                else:
                    items.append(
                        {
                            "ID": uniprot_id,
                            "Sequence": str(seq_records[uniprot_id].seq),
                            "3DI": str(seq_records_struct[uniprot_id].seq),
                        }
                    )
            except KeyError:
                missing += 1

        if missing > 0:
            print(f"Missing {missing} uniprot ids", file=sys.stderr)

        df = pd.DataFrame(items)
        df.to_csv(f"{args.output}/{name}_res_test.csv", index=False)

        foldseek.align(tmp_folder_test, traindb_path, f"{args.output}/{name}_aln.m8")
