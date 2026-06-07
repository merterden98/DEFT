"""Adapter to the foldseek binary.

The single place DEFT shells out to `foldseek`. Every subprocess goes through
`_run`, which raises on a non-zero exit so failures surface here rather than as
a confusing missing-file error several layers up. Alignment parsing and EC
analysis live in `alignment.py`, not here.
"""

import shlex
import subprocess as sp
import tempfile

from Bio import SeqIO

DEFAULT_BINARY = "foldseek"


def _run(cmd: str) -> sp.CompletedProcess:
    """Run a shell command, raising RuntimeError with stderr on failure."""
    proc = sp.run(shlex.split(cmd), capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {proc.returncode}): {cmd}\n{proc.stderr}"
        )
    return proc


def align(
    query: str,
    target: str,
    out_m8: str,
    *,
    cov_mode: int = 2,
    evalue: float = 0.1,
    foldseek: str = DEFAULT_BINARY,
) -> str:
    """Structurally align `query` against `target`, writing alignments to `out_m8`.

    `query` and `target` may each be a directory of structures or a foldseek
    database. Returns `out_m8`.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _run(
            f"{foldseek} easy-search --cov-mode {cov_mode} -e {evalue} "
            f"{query} {target} {out_m8} {tmpdir}/aln"
        )
    return out_m8


def extract_3di_from_db(db_path: str, foldseek: str = DEFAULT_BINARY):
    """Extract (amino-acid, 3Di) SeqRecord dicts from an existing foldseek db.

    foldseek keeps the 3Di sequences in a sibling `<db>_ss` database. To read
    both as FASTA we swap `_ss` into the primary slot, convert, then restore the
    original layout. Returns (aa_records, struct_records, db_path), each dict
    keyed by the bare accession.
    """

    def _records(fasta: str):
        return {
            key.split("_")[0]: value
            for key, value in SeqIO.to_dict(SeqIO.parse(fasta, "fasta")).items()
        }

    _run(f"{foldseek} convert2fasta {db_path} {db_path}.fasta")
    seq_records = _records(f"{db_path}.fasta")

    # back up the amino-acid db, swap the _ss (3Di) db into place
    _run(f"cp {db_path} {db_path}_seq")
    _run(f"cp {db_path}.index {db_path}_seq.index")
    _run(f"mv {db_path}_ss {db_path}")
    _run(f"mv {db_path}_ss.index {db_path}.index")
    _run(f"{foldseek} convert2fasta {db_path} {db_path}_ss.fasta")
    seq_records_struct = _records(f"{db_path}_ss.fasta")

    # restore the original layout
    _run(f"mv {db_path} {db_path}_ss")
    _run(f"mv {db_path}.index {db_path}_ss.index")
    _run(f"mv {db_path}_seq {db_path}")
    _run(f"mv {db_path}_seq.index {db_path}.index")

    return seq_records, seq_records_struct, db_path


def retrieve_3di(structures: str, foldseek: str = DEFAULT_BINARY):
    """Build a foldseek db from a CIF file/directory and extract (aa, 3Di) records.

    The scratch db is created under a `mkdtemp` directory that is intentionally
    left in place: callers reuse the returned db path as an alignment target.
    """
    db_name = hash(structures)
    tmpdir = tempfile.mkdtemp()
    _run(f"{foldseek} createdb {structures} {tmpdir}/{db_name}")
    return extract_3di_from_db(f"{tmpdir}/{db_name}", foldseek=foldseek)
