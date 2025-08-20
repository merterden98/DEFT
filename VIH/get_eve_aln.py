import pandas as pd
import sys
sys.path.append("../")
from cli import CreateDatasetSpecies
from glob import glob
FILE = "recon_microbe.tsv"
TRAIN_DB = "/hpc/group/singhlab/rawdata/afdbMert/clean70_0/clean70_db"
output = "/hpc/group/singhlab/rawdata/afdbMert/vih"

df = pd.read_csv(FILE, sep="\t")

computed = set()

for i, row in df.iterrows():
    try:
        file = f"{output}/{int(row.ncbiid)}"
        if glob(file):
            computed.add(int(row.ncbiid))
    except ValueError:
        pass

for i in df["ncbiid"]:
    species = int(i)
    if species in computed:
        continue
    else:
        create_dataset = CreateDatasetSpecies(species, f"{output}/{species}", TRAIN_DB)
        create_dataset.run()
