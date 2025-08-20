import typing as T
from dataclasses import dataclass
from simple_parsing import ArgumentParser, subparsers
from .create_dataset import CreateDataset, CreateDatasetSpecies
from .predict import Predict
from .train import Train
from .evaluate import Evaluate
from .search import Search
from .annotate import Annotate

@dataclass
class CLI:

    command: T.Union[CreateDataset, Predict, Train, Evaluate, CreateDatasetSpecies, Search, Annotate]

    def run(self):
        self.command.run()

parser = ArgumentParser()
parser.add_arguments(CLI, dest="cli")







