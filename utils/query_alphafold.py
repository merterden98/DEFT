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

import http.client
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional


EBI_PROTEOMES_DIR = "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/"
EBI_PREDICTION_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"

UNIPROT_PROTEOMES_SEARCH = (
    "https://rest.uniprot.org/proteomes/search"
    "?query=organism_id:{tax_id}&format=tsv&fields=upid,protein_count&size=10"
)
UNIPROT_ACCESSION_STREAM = (
    "https://rest.uniprot.org/uniprotkb/stream?query=proteome:{upid}&format=list"
)
UNIPROT_ACCESSION_SEARCH = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=proteome:{upid}&format=list&size=500"
)

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
    if tar_name is not None:
        url = EBI_PROTEOMES_DIR + tar_name
        local_tar = tmp_folder / tar_name
        print(f"Downloading {tar_name} -> {tmp_folder}...")
        _http_download(url, local_tar)
        print(f"Extracting {local_tar.name}")
        with tarfile.open(local_tar) as tf:
            tf.extractall(tmp_folder)
        local_tar.unlink()
        return

    # The EBI /latest/ listing only carries ~46 curated proteomes. For any
    # other taxonomy ID, resolve the reference proteome via UniProt and fetch
    # each AlphaFold prediction individually from the EBI prediction API.
    print(
        f"No tarball at {EBI_PROTEOMES_DIR} for taxonomy_id={taxonomy_id}; "
        f"falling back to per-accession download via UniProt."
    )
    upid, accessions = _uniprot_accessions_for_tax(taxonomy_id)
    if not accessions:
        raise RuntimeError(
            f"No UniProt reference proteome found for taxonomy_id={taxonomy_id}. "
            f"Verify the ID at https://www.uniprot.org/proteomes/."
        )
    print(
        f"Resolved taxonomy_id={taxonomy_id} -> UniProt proteome {upid} "
        f"({len(accessions)} accessions). Fetching CIFs from EBI..."
    )
    _download_uniprot_list(accessions, tmp_folder)


def _uniprot_accessions_for_tax(taxonomy_id: str) -> tuple[Optional[str], List[str]]:
    """Return (proteome_id, accession_list) for the reference proteome of a tax ID.

    Picks the first proteome returned by UniProt for the exact organism_id —
    UniProt orders reference/representative proteomes ahead of redundant ones.
    """
    search_url = UNIPROT_PROTEOMES_SEARCH.format(tax_id=taxonomy_id)
    with urllib.request.urlopen(search_url) as resp:
        rows = resp.read().decode("utf-8", errors="replace").splitlines()
    # Drop the header row; pick the first data row's UPID.
    upid: Optional[str] = None
    for row in rows[1:]:
        if not row.strip():
            continue
        upid = row.split("\t", 1)[0].strip()
        if upid:
            break
    if upid is None:
        return None, []

    accessions = _fetch_uniprot_accessions(upid)
    return upid, accessions


def _fetch_uniprot_accessions(upid: str) -> List[str]:
    """Fetch the accession list for a UniProt proteome.

    The /uniprotkb/stream endpoint occasionally truncates large chunked
    responses (IncompleteRead). Try it with a few retries first; on persistent
    failure, fall back to the paginated /uniprotkb/search endpoint, which is
    chunked per page (500 accessions) and follows the Link: rel="next" header.
    """
    stream_url = UNIPROT_ACCESSION_STREAM.format(upid=upid)
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(stream_url, timeout=120) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return [line.strip() for line in body.splitlines() if line.strip()]
        except (
            urllib.error.URLError,
            http.client.IncompleteRead,
            http.client.RemoteDisconnected,
            ConnectionResetError,
            TimeoutError,
        ) as e:
            last_err = e
            print(
                f"UniProt /stream failed for {upid} "
                f"(attempt {attempt + 1}/3): {type(e).__name__}: {e}",
                file=sys.stderr,
            )
            time.sleep(2 ** attempt)

    print(
        f"Falling back to paginated /search for {upid} after stream failures: {last_err}",
        file=sys.stderr,
    )
    return _fetch_uniprot_accessions_paginated(upid)


def _fetch_uniprot_accessions_paginated(upid: str) -> List[str]:
    url: Optional[str] = UNIPROT_ACCESSION_SEARCH.format(upid=upid)
    accessions: List[str] = []
    while url:
        for attempt in range(3):
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    link = resp.headers.get("Link", "")
                break
            except (
                urllib.error.URLError,
                http.client.IncompleteRead,
                http.client.RemoteDisconnected,
                ConnectionResetError,
                TimeoutError,
            ) as e:
                if attempt == 2:
                    raise
                print(
                    f"UniProt /search retry {attempt + 1}/3 for {url}: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                time.sleep(2 ** attempt)
        accessions.extend(
            line.strip() for line in body.splitlines() if line.strip()
        )
        next_match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = next_match.group(1) if next_match else None
    return accessions


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


_TRANSIENT_NET_ERRORS = (
    urllib.error.URLError,
    http.client.IncompleteRead,
    http.client.RemoteDisconnected,
    http.client.BadStatusLine,
    ConnectionResetError,
    TimeoutError,
)


def _fetch_one(uniprot_id: str, tmp_folder: Path) -> bool:
    api_url = EBI_PREDICTION_API.format(uniprot=uniprot_id)
    last_err: Optional[Exception] = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(api_url, timeout=60) as resp:
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
            # 404/422 are "no prediction available" — don't retry, just skip.
            if e.code in (404, 422):
                return False
            # 5xx and 429 are transient; retry with backoff.
            if e.code >= 500 or e.code == 429:
                last_err = e
            else:
                raise
        except _TRANSIENT_NET_ERRORS as e:
            last_err = e
        time.sleep(1.5 ** attempt)
    print(
        f"Giving up on {uniprot_id} after 4 attempts: "
        f"{type(last_err).__name__}: {last_err}",
        file=sys.stderr,
    )
    return False


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
