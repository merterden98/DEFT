import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from utils import constants
from utils.loader import construct_query, retrieve_model, retrieve_trainer


@dataclass
class Annotate:
    query: str  # cifFile
    model: str
    peft: str
    out: Optional[str] = None

    def run(self):
        return main(self)


def main(args):
    tokenizer, model = retrieve_model(args.model, args.peft)
    dataset = construct_query(args.query, tokenizer, train=False)

    trainer = retrieve_trainer(model, tokenizer, dataset)
    res = trainer.predict(dataset)
    predictions = np.argmax(res.predictions, axis=1)
    predictions = [
        (record["ID"], constants.ec_to_label[pred])
        for record, pred in zip(dataset, predictions)
    ]

    for chain_id, ec in predictions:
        print(f"{chain_id}\t{ec}")

    if args.out:
        pd.DataFrame(predictions, columns=["ID", "EC"]).to_csv(args.out, index=False)

    return predictions
