import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..utils.constants import LEVEL_2_ECS, label_to_ec
from ..utils.loader import construct_dataset, retrieve_model_training, retrieve_trainer

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)


REPO_MODELS_DIR = str(Path(__file__).resolve().parent.parent / "models")


def verify_data_types(model):
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()
    total = sum(dtypes.values())
    for k, v in dtypes.items():
        print(f"{k}, {v}, {v / total}")


def _build_label_to_id(train_csv: str, ec_level: int) -> dict:
    if ec_level == 2:
        return dict(label_to_ec)
    df = pd.read_csv(train_csv, sep=",")
    if "EC" not in df.columns:
        raise ValueError(f"{train_csv} has no EC column; cannot build labels at ec_level={ec_level}")
    keys = sorted(
        {".".join(str(ec).split(".")[:ec_level]) for ec in df["EC"].dropna()}
    )
    return {k: i for i, k in enumerate(keys)}


@dataclass
class Train:
    data: str
    data_eval: str
    save_path: str
    model: str = REPO_MODELS_DIR
    lr: float = 5e-5
    epochs: int = 10
    ec_level: int = 2

    def run(self):
        main(self)


def main(train: Train):
    if train.ec_level not in (2, 4):
        raise ValueError(f"ec_level must be 2 or 4, got {train.ec_level}")

    label_to_id = _build_label_to_id(train.data, train.ec_level)
    num_labels = len(label_to_id)
    print(f"Training at EC level {train.ec_level} with {num_labels} labels")

    tokenizer, model = retrieve_model_training(train.model, num_labels=num_labels)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
        modules_to_save=["classifier"],
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    verify_data_types(model)
    dataset = construct_dataset(
        train.data, tokenizer, train=True,
        label_to_id=label_to_id, ec_level=train.ec_level,
    )
    dataset_eval = construct_dataset(
        train.data_eval, tokenizer, train=True,
        label_to_id=label_to_id, ec_level=train.ec_level,
    )

    os.makedirs(train.save_path, exist_ok=True)
    with open(os.path.join(train.save_path, "labels.json"), "w") as f:
        json.dump({"ec_level": train.ec_level, "label_to_id": label_to_id}, f, indent=2)

    trainer = retrieve_trainer(
        model, tokenizer, dataset, dataset_eval, output_dir=train.save_path
    )
    trainer_results = trainer.train()
    model.save_pretrained(train.save_path)
    trainer.save_model(train.save_path)

    with open(f"{train.save_path}/stats.json", "w") as f:
        json.dump(trainer_results, f)
