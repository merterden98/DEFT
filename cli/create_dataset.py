import typing as T
import pandas as pd
from dataclasses import dataclass
import contextlib
from Bio import SeqIO, SeqRecord
import tempfile
import subprocess as sp
import shlex
from typing import Dict, Tuple
import sys
import os
from pathlib import Path

from utils import query_alphafold


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


def run_foldseek_aln(train_folder, test_folder, output_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_hash = hash(test_folder)
        fseek_base_cmd = f"foldseek easy-search --cov-mode 2 -e 0.1 {test_folder} {train_folder} {output_file} {tmpdir}/aln{tmp_hash}"
        print(fseek_base_cmd)
        proc = sp.Popen(shlex.split(fseek_base_cmd), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()
        return output_file


def run_foldseek(
    folder, ec_data, foldseek="foldseek"
) -> Tuple[Dict[str, SeqRecord.SeqRecord], Dict[str, SeqRecord.SeqRecord], str]:
    pdb_dir_name = hash(folder)
    pdb_dir_name = f"a{pdb_dir_name}"

    for root, _, files in os.walk(folder):
        for fname in files:
            src = os.path.join(root, fname)
            if fname.endswith(".cif.gz"):
                ext = ".cif.gz"
            elif fname.endswith(".cif"):
                ext = ".cif"
            else:
                # AlphaFold proteome tarballs ship .pdb.gz, confidence JSONs,
                # and PAE artifacts alongside the CIFs. foldseek picks up any
                # parseable structure file in the folder, so leaving .pdb.gz
                # behind doubles every entry. Drop everything that isn't a CIF.
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
    with contextlib.nullcontext(tempfile.mkdtemp()) as tmpdir:
        FSEEK_BASE_CMD = f"{foldseek} createdb {folder} {tmpdir}/{pdb_dir_name}"
        proc = sp.Popen(shlex.split(FSEEK_BASE_CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        CMD = f"{foldseek} convert2fasta {tmpdir}/{pdb_dir_name} {tmpdir}/{pdb_dir_name}.fasta"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        seq_records = SeqIO.to_dict(
            SeqIO.parse(f"{tmpdir}/{pdb_dir_name}.fasta", "fasta")
        )
        # Update the keys to only have uniprot id
        seq_records = {key.split("_")[0]: value for key, value in seq_records.items()}
        # create backup of {tmpdir}/{pdb_dir_name}
        CMD = f"cp {tmpdir}/{pdb_dir_name} {tmpdir}/{pdb_dir_name}_seq"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        # create backup of {tmpdir}/{pdb_dir_name}.index
        CMD = f"cp {tmpdir}/{pdb_dir_name}.index {tmpdir}/{pdb_dir_name}_seq.index"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()
        CMD = f"mv {tmpdir}/{pdb_dir_name}_ss {tmpdir}/{pdb_dir_name}"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()
        CMD = f"mv {tmpdir}/{pdb_dir_name}_ss.index {tmpdir}/{pdb_dir_name}.index"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()
        CMD = f"{foldseek} convert2fasta {tmpdir}/{pdb_dir_name} {tmpdir}/{pdb_dir_name}_ss.fasta"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        seq_records_struct = SeqIO.to_dict(
            SeqIO.parse(f"{tmpdir}/{pdb_dir_name}_ss.fasta", "fasta")
        )
        # Update the keys to only have uniprot id
        seq_records_struct = {
            key.split("_")[0]: value for key, value in seq_records_struct.items()
        }

        # Restore the original files
        CMD = f"mv {tmpdir}/{pdb_dir_name} {tmpdir}/{pdb_dir_name}_ss"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        CMD = f"mv {tmpdir}/{pdb_dir_name}.index {tmpdir}/{pdb_dir_name}_ss.index"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        CMD = f"mv {tmpdir}/{pdb_dir_name}_seq {tmpdir}/{pdb_dir_name}"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        CMD = f"mv {tmpdir}/{pdb_dir_name}_seq.index {tmpdir}/{pdb_dir_name}.index"
        proc = sp.Popen(shlex.split(CMD), stdout=sp.PIPE, stderr=sp.PIPE)
        _ = proc.communicate()

        return seq_records, seq_records_struct, f"{tmpdir}/{pdb_dir_name}"


def main_species(args: CreateDatasetSpecies):
    Path(args.output).mkdir(parents=True, exist_ok=True)
    tmp_folder = str(Path(args.output))

    name = f"{args.species}"
    tmp_folder_train = query_alphafold.get_pdb_files([], tmp_folder, name, query=name)
    print(tmp_folder_train)

    seq_records, seq_records_struct, traindb_path = run_foldseek(tmp_folder_train, None)

    # foldseek's fasta keys come from the CIF filenames, so they keep the ".cif"
    # (or ".cif.gz") suffix. read_aln strips that off the alignment Query/Target
    # columns, so we have to strip it here too — otherwise the EC-prefix join in
    # foldseek.assign_predictions silently produces zero rows.
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

    run_foldseek_aln(args.train_db, tmp_folder_train, f"{args.output}/{name}_aln.m8")


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

    seq_records, seq_records_struct, traindb_path = run_foldseek(
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
        run_foldseek_aln(
            args.train_db, tmp_folder_train, f"{args.output}/{name}_aln.m8"
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
        seq_records, seq_records_struct, _ = run_foldseek(tmp_folder_test, ec_data)

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

        run_foldseek_aln(traindb_path, tmp_folder_test, f"{args.output}/{name}_aln.m8")
