"""Download AlphaFold structures without any GCP / gsutil / BigQuery setup.

Two access paths are supported:

1. **By taxonomy / species ID** — used by `create-dataset-species` and
   `easy-predict`. AlphaFold proteome tarballs are mirrored on the EBI
   FTP site at `https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/`
   under names like `UP000002438_208964_PSEAE_v6.tar`. We scrape the
   directory listing to find the tarball matching a taxonomy ID and
   stream it over HTTPS.

2. **By UniProt accession list** — used by `create-dataset --file`. Each
   accession resolves to a CIF via the EBI AlphaFold prediction API
   (`https://alphafold.ebi.ac.uk/api/prediction/{uniprot}`), which returns
   a versioned `cifUrl`. No auth, no BigQuery join, no `gsutil`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional


EBI_PROTEOMES_DIR = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/"
EBI_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"

DEFAULT_WORKERS = 16


def get_pdb_files(
    uniprot_ids: list,
    download_dir: str,
    tmp_prefix: str,
    query: Optional[str] = None,
) -> str:
    """Download AlphaFold structures into a fresh subdirectory and return its path.

    If `query` is provided, it's treated as a taxonomy ID and the matching
    proteome tarballs are pulled from the public AlphaFold bucket. Otherwise
    `uniprot_ids` is used and each accession is fetched individually from EBI.
    """
    tmp_folder = Path(download_dir) / f"{tmp_prefix}_{uuid.uuid4()}"
    tmp_folder.mkdir(parents=True, exist_ok=True)

    if query:
        taxonomy_id = query.lstrip()
        if taxonomy_id.startswith("taxonomy_id:"):
            taxonomy_id = taxonomy_id.split(":", 1)[1]
        _download_proteome(taxonomy_id, tmp_folder)
        return str(tmp_folder)

    if uniprot_ids:
        _download_uniprot_list(uniprot_ids, tmp_folder)
    return str(tmp_folder)


def _download_proteome(taxonomy_id: str, tmp_folder: Path) -> None:
    print(f"Looking up EBI proteome tarball for taxonomy_id={taxonomy_id}...")
    tar_name = _find_proteome_tar(taxonomy_id)
    if tar_name is None:
        raise RuntimeError(
            f"No AlphaFold proteome tarball found for taxonomy_id={taxonomy_id} "
            f"at {EBI_PROTEOMES_DIR}. Verify the ID at https://alphafold.ebi.ac.uk/."
        )
    url = EBI_PROTEOMES_DIR + tar_name
    local_tar = tmp_folder / tar_name
    print(f"Downloading {tar_name} -> {tmp_folder}...")
    _http_download(url, local_tar)
    print(f"Extracting {local_tar.name}")
    with tarfile.open(local_tar) as tf:
        tf.extractall(tmp_folder)
    local_tar.unlink()


_PROTEOME_RE = re.compile(
    r'href="(UP\d+_{tax_id}_[A-Z0-9]+_v(\d+)\.tar)"'
)


def _find_proteome_tar(taxonomy_id: str) -> Optional[str]:
    """Pick the highest-version `UP*_{taxonomy_id}_*_vN.tar` from the EBI listing."""
    with urllib.request.urlopen(EBI_PROTEOMES_DIR) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    pattern = re.compile(
        rf'href="(UP\d+_{re.escape(taxonomy_id)}_[A-Z0-9]+_v(\d+)\.tar)"'
    )
    candidates = [(int(m.group(2)), m.group(1)) for m in pattern.finditer(html)]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _download_uniprot_list(uniprot_ids: Iterable[str], tmp_folder: Path) -> None:
    ids: List[str] = [u.strip() for u in uniprot_ids if u.strip()]
    print(f"Downloading {len(ids)} AlphaFold structures from EBI to {tmp_folder}...")
    missing = 0
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, uid, tmp_folder): uid for uid in ids
        }
        for fut in as_completed(futures):
            if not fut.result():
                missing += 1
    if missing:
        print(f"Warning: {missing} of {len(ids)} structures could not be fetched.", file=sys.stderr)


def _fetch_one(uniprot_id: str, tmp_folder: Path) -> bool:
    api_url = EBI_PREDICTION_API.format(uniprot=uniprot_id)
    try:
        with urllib.request.urlopen(api_url) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if not payload:
            return False
        cif_url = payload[0].get("cifUrl")
        if not cif_url:
            return False
        out = tmp_folder / Path(urllib.parse.urlparse(cif_url).path).name
        _http_download(cif_url, out)
        return True
    except urllib.error.HTTPError as e:
        if e.code in (404, 422):
            return False
        raise


def _http_download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp_dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp_dest, dest)
