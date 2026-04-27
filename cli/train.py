import json
from dataclasses import dataclass
from pathlib import Path
from utils.loader import construct_dataset, retrieve_model_training, retrieve_trainer

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


@dataclass
class Train:
    data: str
    data_eval: str
    save_path: str
    model: str = REPO_MODELS_DIR
    lr: float = 5e-5
    epochs: int = 10

    def run(self):
        main(self)


def main(train: Train):
    tokenizer, model = retrieve_model_training(train.model)

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
    dataset = construct_dataset(train.data, tokenizer, train=True)
    dataset_eval = construct_dataset(train.data_eval, tokenizer, train=True)
    trainer = retrieve_trainer(
        model, tokenizer, dataset, dataset_eval, output_dir=train.save_path
    )
    trainer_results = trainer.train()
    model.save_pretrained(train.save_path)
    trainer.save_model(train.save_path)

    with open(f"{train.save_path}/stats.json", "w") as f:
        json.dump(trainer_results, f)
