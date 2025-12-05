import pandas as pd
import typing as T
import numpy as np
import torch
import tempfile
from dataclasses import dataclass
from argparse import ArgumentParser
from transformers import EsmTokenizer, EsmForSequenceClassification
from transformers import BatchEncoding
from transformers import Trainer, TrainingArguments
from utils import constants
from utils import foldseek
from utils.loader import (
    construct_query,
    retrieve_model,
    retrieve_trainer,
    construct_dataset_from_df,
)


@dataclass
class Annotate:
    query: str  # cifFile
    model: str
    peft: str

    def run(self):
        return main(self)


def main(args):
    # Load the model and tokenizer and convert the query pdb into a dataset
    tokenizer, model = retrieve_model(args.model, args.peft)
    dataset = construct_query(args.query, tokenizer, train=False)

    trainer = retrieve_trainer(model, tokenizer, dataset)
    res = trainer.predict(dataset)
    predictions = np.argmax(res.predictions, axis=1)
    predictions = [
        (i["ID"], "", constants.ec_to_label[pred])
        for (i, pred) in zip(dataset, predictions)
    ]

    for i in predictions:
        print(i)

    return predictions
