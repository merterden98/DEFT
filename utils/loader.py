import typing as T
import pandas as pd
import evaluate
from transformers import EsmTokenizer, EsmForSequenceClassification, EsmConfig
from transformers import BatchEncoding
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    SchedulerType,
)
from datasets import Dataset
from .constants import LEVEL_2_ECS as LABELS, label_to_ec
from safetensors.torch import load_file as safe_load_file

# import linear schedule with warmup
from peft import (
    get_peft_model,
    PeftConfig,
    PeftModel,
)
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


def construct_query(
    query: str, tokenizer: EsmTokenizer, train: bool = False
) -> Dataset:
    aa_records, struct_records, path = retrieve_3di(query)

    data = pd.DataFrame(
        [
            {
                "ID": k,
                "Sequence": str(aa_records[k].seq),
                "3DI": str(struct_records[k].seq),
            }
            for k in aa_records.keys()
        ]
    )

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


def construct_dataset(
    file_path: str, tokenizer: EsmTokenizer, train: bool = False
) -> Dataset:
    data = pd.read_csv(file_path, sep=",")
    return construct_dataset_from_df(data, tokenizer, train)


def construct_dataset_from_df(
    data: pd.DataFrame, tokenizer: EsmTokenizer, train: bool = False
) -> Dataset:
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
    import os

    tokenizer_source = (
        peft_path
        if os.path.exists(os.path.join(peft_path, "tokenizer_config.json"))
        else model_path
    )

    def _infer_num_labels_from_adapter(adapter_dir: str) -> int | None:
        try:
            # Prefer top-level adapter weights; fallback to checkpoint subfolder
            cand_files = [
                os.path.join(adapter_dir, "adapter_model.safetensors"),
                os.path.join(
                    adapter_dir, "checkpoint-14500", "adapter_model.safetensors"
                ),
            ]
            for f in cand_files:
                if os.path.exists(f):
                    sd = safe_load_file(f)
                    # Try common key variants
                    for k in sd.keys():
                        if k.endswith(
                            "classifier.modules_to_save.default.out_proj.weight"
                        ):
                            return sd[k].shape[0]
            return None
        except Exception:
            return None

    config = EsmConfig.from_pretrained(model_path)
    inferred_labels = _infer_num_labels_from_adapter(peft_path)
    config.num_labels = inferred_labels if inferred_labels is not None else len(LABELS)
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_source)
    base_model: EsmForSequenceClassification = (
        EsmForSequenceClassification.from_pretrained(model_path, config=config)
    )  # type: ignore

    try:
        peft_conf = PeftConfig.from_pretrained(peft_path)
        peft_wrapped = get_peft_model(base_model, peft_conf)
        peft_wrapped.load_adapter(peft_path, adapter_name="default")
        peft_wrapped.set_adapter("default")
        return tokenizer, peft_wrapped
    except Exception as e:
        print(f"get_peft_model failed: {e}; using PeftModel.from_pretrained")
        peft_model = PeftModel.from_pretrained(base_model, peft_path)
        return tokenizer, peft_model


def retrieve_model_training(model_path, quantization_config=None):
    # Check if model_path is a local directory with model weights
    import os

    if os.path.exists(model_path) and os.path.isdir(model_path):
        # Check if model weights exist in the directory
        weight_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        ]
        has_weights = any(
            os.path.exists(os.path.join(model_path, f)) for f in weight_files
        )

        if has_weights:
            # Use local model
            config = EsmConfig.from_pretrained(model_path)
            config.num_labels = len(LABELS)
            tokenizer = EsmTokenizer.from_pretrained(model_path)
            model: EsmForSequenceClassification = (
                EsmForSequenceClassification.from_pretrained(model_path, config=config)
            )  # type: ignore
        else:
            # Load config from local path to get base model name
            config = EsmConfig.from_pretrained(model_path)
            base_model = config._name_or_path
            print(
                f"Model weights not found in {model_path}, using base model: {base_model}"
            )
            config.num_labels = len(LABELS)
            tokenizer = EsmTokenizer.from_pretrained(base_model)
            model: EsmForSequenceClassification = (
                EsmForSequenceClassification.from_pretrained(base_model, config=config)
            )  # type: ignore
    else:
        # Use the base model from Hugging Face
        base_model = "westlake-repl/SaProt_650M_AF2"
        print(f"Model path {model_path} not found, using base model: {base_model}")
        config = EsmConfig.from_pretrained(base_model)
        config.num_labels = len(LABELS)
        tokenizer = EsmTokenizer.from_pretrained(base_model)
        model: EsmForSequenceClassification = (
            EsmForSequenceClassification.from_pretrained(base_model, config=config)
        )  # type: ignore

    return tokenizer, model


def retrieve_trainer(
    model, tokenizer, train_dataset=None, eval_dataset=None, output_dir="./results"
):
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(pred):
        predictions, labels = pred
        preds = predictions.argmax(-1)
        return acc_metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to=["wandb"],
        logging_steps=100,
        evaluation_strategy="steps",
        num_train_epochs=10,
        label_names=["labels"],
        eval_steps=0.1,
        learning_rate=1e-4,
        save_strategy="epoch",
        save_total_limit=10,
        fp16=True,
        optim="adafactor",
        # save every epoch
        save_steps=1,
        lr_scheduler_type=SchedulerType.REDUCE_ON_PLATEAU,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        # optimizers=(optimizer, lr_scheduler)
    )
    return trainer
