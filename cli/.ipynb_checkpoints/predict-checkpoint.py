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
from utils.loader import construct_dataset, retrieve_model, retrieve_trainer, retrieve_model_training

@dataclass
class Predict:
    data : str
    align : str
    model : str
    peft : str
    train_csv : str

    def run(self):
        return main(self)



def eve_filter(predictions, dataset, align, train_csv):
    aln = foldseek.read_aln(align)
    aln = foldseek.assign_predictions(aln, dataset, predictions, train_csv)
    return aln





def main(args):
    tokenizer, model = retrieve_model(args.model, args.peft)
    dataset = construct_dataset(args.data, tokenizer, train=False)
    trainer = retrieve_trainer(model, tokenizer, dataset)
    res = trainer.predict(dataset)
    predictions = np.argmax(res.predictions, axis=1)
    predictions = [(i["ID"], "", constants.ec_to_label[pred]) for (i, pred) in zip(dataset, predictions)]
    aln = eve_filter(predictions, dataset, args.align, args.train_csv)
    return aln

    
