from google.cloud import bigquery
import os
import pandas as pd
import tempfile
import uuid



os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/hpc/home/me196/projects/EnzymeClassification/dataset_scripts/key.json"

client = bigquery.Client()
alphafold_db = "bigquery-public-data.deepmind_alphafold"

TAXON_DOWNLOAD_BUCK = "gs://public-datasets-deepmind-alphafold-v4/proteomes/proteome-tax_id-{taxonomy_id}-*_v4.tar"


config = bigquery.LoadJobConfig(
schema=[
    bigquery.SchemaField("uniprotAccession", "STRING"),
], write_disposition="WRITE_TRUNCATE")

def get_pdb_files(uniprot_ids: list, download_dir : str, tmp_prefix : str, query=None) -> str:

    tmp_folder = f"{download_dir}/{tmp_prefix}_{uuid.uuid4()}"
    # Make directory
    os.system(f"mkdir {tmp_folder}")


    # lstrip query and see if starts with taxonomy_id:
    try:
        if query:
            if query.lstrip().startswith("taxonomy_id:"):
                taxonomy_id = query.lstrip().split(":")[1]
                # Download the proteome using gsutil
                os.system(f"gsutil -m cp {TAXON_DOWNLOAD_BUCK.format(taxonomy_id=taxonomy_id)} {tmp_folder}")
                return tmp_folder
    except:
        print("Could not download proteome, downloading individual files instead")


    # Upload the list of uniprot_ids to bigquery temporarily

    df = pd.DataFrame(uniprot_ids, columns=["uniprotAccession"])
    job_result = client.load_table_from_dataframe(df, "merden01.uniprot_ids", job_config=config).result()



    QUERY = """

    WITH file_rows AS (
        WITH file_cols AS (
            SELECT
                CONCAT(entryID, '-model_v4.cif') as m,
                CONCAT(entryID, '-predicted_aligned_error_v4.json') as p
            FROM bigquery-public-data.deepmind_alphafold.metadata as g
            INNER JOIN `mucin-407221.merden01.uniprot_ids` as b ON g.uniprotAccession = b.uniprotAccession
        )
        SELECT * FROM file_cols UNPIVOT (files FOR filetype IN (m, p))
    )
    SELECT CONCAT('gs://public-datasets-deepmind-alphafold-v4/', files) AS files
    FROM file_rows


    """

    query_job = client.query(QUERY)
    results = query_job.result()
    gs_bucket_files = [row.files for row in results if row.files.endswith(".cif")]
    print("Downloading Files to tmp folder", tmp_folder, "...")
    try:
        # Download files
        download_pdb_files(gs_bucket_files, tmp_folder)
    except OSError as e:
        print("OsError", e)

    return tmp_folder


def download_pdb_files(gs_bucket_files: list, output_dir: str) -> None:

    f = tempfile.NamedTemporaryFile("w", delete=False)
    # Create a temporary file and write all of the gs_bucket_files
    for file in gs_bucket_files:
        f.write(f"{file}\n")
    f.flush()
    print("Downloading files...", f.name, "...")
    os.system(f"cat {f.name} | TMPDIR=/tmp/ gsutil -m cp -I {output_dir}")

