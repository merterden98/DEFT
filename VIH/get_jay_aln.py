import pandas as pd
import sys
sys.path.append("../")
from cli import CreateDatasetSpecies
from glob import glob
FILE = "recon_microbe.tsv"
TRAIN_DB = "/hpc/group/singhlab/rawdata/afdbMert/clean70_0/clean70_db"
output = "/hpc/group/singhlab/rawdata/afdbMert/jay"


ids = [89584, 211586, 1028307, 592022, 1131758,666685]

computed = set()

for i in ids:
    try:
        file = f"{output}/{int(i)}"
        if glob(file):
            computed.add(int(i))
    except ValueError:
        pass

for i in ids:
    species = int(i)
    if species in computed:
        continue
    else:
        create_dataset = CreateDatasetSpecies(species, f"{output}/{species}", TRAIN_DB)
        create_dataset.run()
