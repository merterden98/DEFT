import numpy as np
from dataclasses import dataclass
from utils import constants
from utils import foldseek
from utils.loader import (
    construct_dataset,
    retrieve_model,
    retrieve_trainer,
)


@dataclass
class Predict:
    data: str
    align: str
    model: str
    peft: str
    train_csv: str
    outfile: str

    def run(self):
        return main(self)


def eve_filter(predictions, dataset, align, train_csv):
    aln = foldseek.read_aln(align)
    aln = foldseek.assign_predictions(aln, dataset, predictions, train_csv)
    return aln


def main(args):
    tokenizer, model = retrieve_model(args.model, args.peft)
    dataset = construct_dataset(args.data, tokenizer, train=False)
    trainer = retrieve_trainer(model, tokenizer, eval_dataset=dataset)
    res = trainer.predict(dataset)
    predictions = np.argmax(res.predictions, axis=1)
    predictions = [
        (i["ID"], "", constants.ec_to_label[pred])
        for (i, pred) in zip(dataset, predictions)
    ]
    aln = eve_filter(predictions, dataset, args.align, args.train_csv)
    aln.to_csv(args.outfile, index=False)
    return aln
