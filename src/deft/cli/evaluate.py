import os
from dataclasses import dataclass
from transformers import EsmTokenizer
from ..utils import foldseek
from ..utils.loader import (
    construct_dataset,
    retrieve_model,
    predict_ec,
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
        predictions = predict_ec(model, tokenizer, dataset)

    aln, accuracy, eval_metrics = eve_filter(
        predictions,
        dataset,
        args.align,
        args.train_csv,
        filter_by_prediction_prefix=not skip_filter,
    )
    aln.to_csv(args.out, index=False)
    return eval_metrics
