import pandas as pd
from dataclasses import dataclass
from typing import Optional
from ..utils.loader import construct_query, retrieve_model, predict_ec


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

    predictions = predict_ec(model, tokenizer, dataset)

    for chain_id, ec in predictions:
        print(f"{chain_id}\t{ec}")

    if args.out:
        pd.DataFrame(predictions, columns=["ID", "EC"]).to_csv(args.out, index=False)

    return predictions
