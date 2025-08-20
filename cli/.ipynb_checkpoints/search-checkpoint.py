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
from utils.loader import construct_query, retrieve_model, retrieve_trainer, construct_dataset_from_df


@dataclass
class Search:
    query : str # cifFile
    db : str
    model : str
    peft : str

    def run(self):
        return main(self)



def eve_filter(predictions, dataset, align, predictions_db):
    aln = foldseek.read_aln(align)
    aln = foldseek.restrict_aln(aln, predictions, dataset, predictions_db)


    return aln





def main(args):

    # Load the model and tokenizer and convert the query pdb into a dataset
    tokenizer, model = retrieve_model(args.model, args.peft)
    dataset = construct_query(args.query, tokenizer, train=False)


    # Take the foldseek db and convert it into a dataset
    seq_records, struct_records, _ = foldseek.extract_3di_from_db(args.db)
    db_dataset = pd.DataFrame([{"ID": k, "Sequence": str(seq_records[k].seq), "3DI": str(struct_records[k].seq)} for k in seq_records.keys()])
    dataset_db = construct_dataset_from_df(db_dataset, tokenizer, train=False)



    # Prealign the query and db: TODO: This can be optimized
    named_temp_file = tempfile.NamedTemporaryFile(delete=False).name
    aln = foldseek.run_foldseek_aln(args.db, args.query, named_temp_file)
    trainer = retrieve_trainer(model, tokenizer, dataset)
    res = trainer.predict(dataset)
    predictions = np.argmax(res.predictions, axis=1)
    predictions = [(i["ID"], "", constants.ec_to_label[pred]) for (i, pred) in zip(dataset, predictions)]


    # Get predictions for the db
    res = trainer.predict(dataset_db)
    predictions_db = np.argmax(res.predictions, axis=1)
    predictions_db = [(i["ID"], "", constants.ec_to_label[pred]) for (i, pred) in zip(dataset_db, predictions_db)]

    aln = eve_filter(predictions, dataset, aln, predictions_db)
    aln.to_csv("aln_filtered.m8")
    return aln
