from typing import Optional
import os
import shutil

import typer


DEFAULT_MODEL = "westlake-repl/SaProt_650M_AF2"


app = typer.Typer(help="DEFT command line interface", rich_markup_mode="ascii")


@app.command("create-dataset")
def create_dataset(
    file: str = typer.Option(
        ...,
        "--file",
        help="Path to input file with UniProt IDs, one per line.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        help="Output directory for generated dataset files.",
    ),
    ec: Optional[str] = typer.Option(
        None,
        "--ec",
        help="Optional path to EC annotation file (tab-separated UniProt ID and EC).",
    ),
    test_file: Optional[str] = typer.Option(
        None,
        "--test-file",
        help="Optional file with UniProt IDs for test set.",
    ),
    test_ec: Optional[str] = typer.Option(
        None,
        "--test-ec",
        help="Optional EC annotation file for test set.",
    ),
    train_db: Optional[str] = typer.Option(
        None,
        "--train-db",
        help="Optional training foldseek database path (required when mode=='test').",
    ),
    mode: str = typer.Option(
        "train",
        "--mode",
        help="Dataset generation mode: 'train' or 'test'.",
        case_sensitive=False,
    ),
) -> None:
    """Create DEFT training or test datasets from UniProt IDs and (optionally) EC labels."""
    from .create_dataset import CreateDataset

    args = CreateDataset(
        file=file,
        output=output,
        ec=ec,
        test_file=test_file,
        test_ec=test_ec,
        train_db=train_db,
        mode=mode,
    )
    args.run()


@app.command("create-dataset-species")
def create_dataset_species(
    species: str = typer.Option(
        ...,
        "--species",
        help="Species identifier.",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        help="Output directory for generated dataset files.",
    ),
    train_db: str = typer.Option(
        ...,
        "--train-db",
        help="Training foldseek database path.",
    ),
) -> None:
    """Create DEFT datasets for a given species using AlphaFold structures."""
    from .create_dataset import CreateDatasetSpecies

    args = CreateDatasetSpecies(
        species=species,
        output=output,
        train_db=train_db,
    )
    args.run()


@app.command("predict")
def predict(
    data: str = typer.Option(
        ...,
        "--data",
        help="CSV file with DEFT input data.",
    ),
    align: str = typer.Option(
        ...,
        "--align",
        help="Foldseek alignment (.m8) file.",
    ),
    peft: str = typer.Option(
        ...,
        "--peft",
        help="Directory containing the PEFT adapter weights.",
    ),
    train_csv: str = typer.Option(
        ...,
        "--train-csv",
        "--train_csv",
        help="Training CSV file used to build the alignment database.",
    ),
    outfile: str = typer.Option(
        ...,
        "--outfile",
        "--out",
        help="Output CSV file for predictions.",
    ),
) -> None:
    """Run DEFT predictions on a dataset and alignment, writing results to a CSV file."""
    from .predict import Predict

    model = DEFAULT_MODEL

    args = Predict(
        data=data,
        align=align,
        model=model,
        peft=peft,
        train_csv=train_csv,
        outfile=outfile,
    )
    args.run()


@app.command("train")
def train(
    data: str = typer.Option(
        ...,
        "--data",
        help="Training CSV file.",
    ),
    data_eval: str = typer.Option(
        ...,
        "--data-eval",
        help="Evaluation CSV file.",
    ),
    save_path: str = typer.Option(
        ...,
        "--save-path",
        help="Directory to save trained model and statistics.",
    ),
    lr: float = typer.Option(
        5e-5,
        "--lr",
        help="Learning rate.",
    ),
    epochs: int = typer.Option(
        10,
        "--epochs",
        help="Number of training epochs.",
    ),
) -> None:
    """Fine-tune the DEFT model with PEFT on a labelled dataset."""
    from .train import Train

    args = Train(
        data=data,
        data_eval=data_eval,
        save_path=save_path,
        lr=lr,
        epochs=epochs,
    )
    args.run()


@app.command("evaluate")
def evaluate(
    data: str = typer.Option(
        ...,
        "--data",
        help="CSV file with labelled evaluation data.",
    ),
    align: str = typer.Option(
        ...,
        "--align",
        help="Foldseek alignment (.m8) file.",
    ),
    train_csv: str = typer.Option(
        ...,
        "--train-csv",
        "--train_csv",
        help="Training CSV file used to build the alignment database.",
    ),
    peft: str = typer.Option(
        ...,
        "--peft",
        help="Directory containing the PEFT adapter weights.",
    ),
    out: str = typer.Option(
        ...,
        "--out",
        help="Output CSV file for evaluation alignments with EC labels.",
    ),
    skip_prediction_filter: bool = typer.Option(
        False,
        "--skip-prediction-filter/--no-skip-prediction-filter",
        help="If set, skip prediction-based filtering and only add EC labels from training data.",
    ),
) -> None:
    """Evaluate DEFT predictions against labelled data and write metrics."""
    from .evaluate import Evaluate

    model = DEFAULT_MODEL

    args = Evaluate(
        data=data,
        align=align,
        train_csv=train_csv,
        model=model,
        peft=peft,
        out=out,
        skip_prediction_filter=skip_prediction_filter,
    )
    args.run()


@app.command("search")
def search(
    query: str = typer.Option(
        ...,
        "--query",
        help="Query CIF file.",
    ),
    db: str = typer.Option(
        ...,
        "--db",
        help="Foldseek database path.",
    ),
    peft: str = typer.Option(
        ...,
        "--peft",
        help="Directory containing the PEFT adapter weights.",
    ),
) -> None:
    """Search a structural database for matches to a query structure using DEFT predictions."""
    from .search import Search

    model = DEFAULT_MODEL

    args = Search(
        query=query,
        db=db,
        model=model,
        peft=peft,
    )
    args.run()


@app.command("annotate")
def annotate(
    query: str = typer.Option(
        ...,
        "--query",
        help="Query CIF file.",
    ),
    peft: str = typer.Option(
        ...,
        "--peft",
        help="Directory containing the PEFT adapter weights.",
    ),
) -> None:
    """Annotate a query structure with EC predictions using DEFT."""
    from .annotate import Annotate

    model = DEFAULT_MODEL

    args = Annotate(
        query=query,
        model=model,
        peft=peft,
    )
    args.run()


@app.command("download")
def download(
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help="Cache directory for DEFT data (defaults to ~/.deft_cache or $DEFT_CACHE).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-download even if files already exist.",
    ),
) -> None:
    """Download DEFT model weights and data files from Zenodo."""
    from .download import Download

    args = Download(
        cache_dir=cache_dir,
        force=force,
    )
    args.run()


@app.command("easy-predict")
def easy_predict(
    species_id: int = typer.Option(
        ...,
        "--species-id",
        help="NCBI taxonomy ID (or other species identifier).",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        help="Directory to store intermediate and output files.",
    ),
    cache_dir: Optional[str] = typer.Option(
        None,
        "--cache-dir",
        help="Optional cache directory for DEFT data (defaults to ~/.deft_cache or $DEFT_CACHE).",
    ),
) -> None:
    """High-level convenience command that downloads DEFT data and runs dataset creation plus prediction."""
    from .easy_predict import EasyPredict

    args = EasyPredict(
        species_id=species_id,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    args.run()


@app.command("gcp-check")
def gcp_check() -> None:
    """Check that GOOGLE_APPLICATION_CREDENTIALS and gsutil are correctly configured."""
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        typer.echo(
            "GOOGLE_APPLICATION_CREDENTIALS is not set.\n"
            "Set it to the path of your GCP service account JSON key file, e.g.:\n"
            '  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"'
        )
        raise typer.Exit(code=1)

    if not os.path.exists(creds):
        typer.echo(
            f"GOOGLE_APPLICATION_CREDENTIALS is set to '{creds}', "
            "but that file does not exist.\n"
            "Make sure the path is correct and readable."
        )
        raise typer.Exit(code=1)

    if shutil.which("gsutil") is None:
        typer.echo(
            "`gsutil` was not found in your PATH.\n"
            "Install the Google Cloud SDK and ensure `gsutil` is available, "
            "then try again.\n"
            "See the README section on GCP / AlphaFold setup for details."
        )
        raise typer.Exit(code=1)

    typer.echo("GCP credentials and gsutil appear to be correctly configured.")


__all__ = ["app"]
