import typing as T
import torch
import pandas as pd
import evaluate
from transformers import EsmTokenizer, EsmForSequenceClassification, EsmConfig
from transformers import BatchEncoding
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, SchedulerType
from accelerate import Accelerator
from datasets import Dataset
from .constants import LEVEL_2_ECS as LABELS, ec_to_label, label_to_ec
from torch.optim import AdamW
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
# import linear schedule with warmup
from torch.optim.lr_scheduler import LinearLR
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PeftModel
from .foldseek import retrieve_3di

REQUIRED_COLUMNS = ["ID", "Sequence", "3DI"]

def get_nb_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"trainable params: {trainable_params: ,} || all params: {all_param: ,} || trainable%: {100 * trainable_params / all_param}"
    )

def construct_query(query: str, tokenizer : EsmTokenizer, train : bool=False) -> Dataset:
    aa_records, struct_records, path = retrieve_3di(query)
    
    data = pd.DataFrame([{
        "ID": k,
        "Sequence": str(aa_records[k].seq),
        "3DI": str(struct_records[k].seq)
    } for k in aa_records.keys()])

    if not all([col in data.columns for col in REQUIRED_COLUMNS]):
        raise ValueError(f"File must contain columns {REQUIRED_COLUMNS}")

    dataset = Dataset.from_pandas(data)

    def _process_row(row: T.Dict[str, T.Any]) -> BatchEncoding:
        # create a merged column of sequence and 3DI where 3DI is lower case and interspereced with sequence
        di_aa = zip(row["Sequence"], row["3DI"].lower())
        di_aa = "".join([f"{aa}{di}" for aa, di in di_aa])
        if train:
            inputs = tokenizer(di_aa, max_length=1024, truncation=True)
            ec_2 = ".".join(row["EC"].split(".")[:2])
            return {**inputs, "labels": label_to_ec[ec_2]}
        return tokenizer(di_aa, padding="max_length", truncation=True, max_length=1024)

    dataset = dataset.map(_process_row)
    return dataset


def construct_dataset(file_path: str, tokenizer : EsmTokenizer, train : bool=False) -> Dataset:
    data = pd.read_csv(file_path, sep=",")
    return construct_dataset_from_df(data, tokenizer, train)


def construct_dataset_from_df(data : pd.DataFrame , tokenizer : EsmTokenizer, train : bool=False) -> Dataset:

    if not all([col in data.columns for col in REQUIRED_COLUMNS]):
        raise ValueError(f"File must contain columns {REQUIRED_COLUMNS}")

    dataset = Dataset.from_pandas(data)

    def _process_row(row: T.Dict[str, T.Any]) -> BatchEncoding:
        # create a merged column of sequence and 3DI where 3DI is lower case and interspereced with sequence
        di_aa = zip(row["Sequence"], row["3DI"].lower())
        di_aa = "".join([f"{aa}{di}" for aa, di in di_aa])
        if train:
            inputs = tokenizer(di_aa, max_length=1024, truncation=True)
            ec_2 = ".".join(row["EC"].split(".")[:2])
            return {**inputs, "labels": label_to_ec[ec_2]}
        return tokenizer(di_aa, padding="max_length", truncation=True, max_length=1024)

    dataset = dataset.map(_process_row)
    return dataset


def retrieve_model(model_path: str, peft_path):
    config = EsmConfig.from_pretrained(model_path)
    config.num_labels = len(LABELS)
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model : EsmForSequenceClassification = EsmForSequenceClassification.from_pretrained(model_path, config=config) # type: ignore
    peft_model = PeftModel.from_pretrained(model, peft_path)
    return tokenizer, peft_model

def retrieve_model_training(model_path, quantization_config=None):
    config = EsmConfig.from_pretrained(model_path)
    config.num_labels = len(LABELS)
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model : EsmForSequenceClassification = EsmForSequenceClassification.from_pretrained(model_path, config=config) # type: ignore

    return tokenizer, model


def retrieve_trainer(model, tokenizer, train_dataset=None, eval_dataset=None, output_dir="./results"):
    acc_metric = evaluate.load("accuracy")
    def compute_metrics(pred):
        predictions ,labels = pred
        preds = predictions.argmax(-1)
        return acc_metric.compute(predictions=preds, references=labels)


    training_args = TrainingArguments(output_dir=output_dir, report_to=["wandb"], logging_steps=100, evaluation_strategy="steps", num_train_epochs=5, label_names=["labels"], eval_steps=0.1, learning_rate=3e-5, save_strategy="epoch", save_total_limit=1, fp16=True, optim="adafactor", lr_scheduler_type=SchedulerType.REDUCE_ON_PLATEAU)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
        # optimizers=(optimizer, lr_scheduler)
    )
    return trainer
