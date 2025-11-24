import os
import pandas as pd
import numpy as np
import typing as T
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
    construct_dataset,
    retrieve_model,
    retrieve_trainer,
    retrieve_model_training,
)


@dataclass
class Evaluate:
    data: str
    align: str
    train_csv: str
    model: str
    peft: str
    out: str
    skip_prediction_filter: bool = False

    def run(self):
        return main(self)


def eve_filter(
    predictions,
    dataset,
    align,
    train_csv,
    filter_by_prediction_prefix: bool = True,
):
    aln = foldseek.read_aln(align)
    aln, accuracy, eval_metrics = foldseek.add_ec_data(
        aln,
        dataset,
        predictions,
        train_csv,
        filter_by_prediction_prefix=filter_by_prediction_prefix,
    )
    return aln, accuracy, eval_metrics


def main(args):
    skip_filter = getattr(args, "skip_prediction_filter", False)

    if skip_filter:
        peft_path = getattr(args, "peft", "")
        tokenizer_source = (
            peft_path
            if peft_path
            and os.path.exists(os.path.join(peft_path, "tokenizer_config.json"))
            else args.model
        )
        tokenizer = EsmTokenizer.from_pretrained(tokenizer_source)
        model = None
    else:
        tokenizer, model = retrieve_model(args.model, args.peft)

    dataset = construct_dataset(args.data, tokenizer, train=True)

    predictions = None
    if not skip_filter:
        trainer = retrieve_trainer(model, tokenizer, dataset)
        res = trainer.predict(dataset)
        prediction_labels = np.argmax(res.predictions, axis=1)
        predictions = [
            (record["ID"], record["EC"], constants.ec_to_label[label_idx])
            for record, label_idx in zip(dataset, prediction_labels)
        ]

    aln, accuracy, eval_metrics = eve_filter(
        predictions,
        dataset,
        args.align,
        args.train_csv,
        filter_by_prediction_prefix=not skip_filter,
    )
    aln.to_csv(args.out, index=False)
    return eval_metrics
