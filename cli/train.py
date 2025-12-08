import json
import torch
from dataclasses import dataclass
from utils.loader import construct_dataset, retrieve_model_training, retrieve_trainer

# from transformers.trainer_utils import TrainOutput
from transformers import BitsAndBytesConfig

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)


def verify_data_types(model):
    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(f"{k}, {v}, {v / total}")


@dataclass
class Train:
    data: str
    data_eval: str
    save_path: str
    lr: float = 5e-5
    epochs: int = 10

    def run(self):
        main(self)


def main(train: Train):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer, model = retrieve_model_training(
        "models/", quantization_config=quantization_config
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
        modules_to_save=["classifier"],  # also try ["dense"
        lora_dropout=0.1,
        bias="none",  # or "all" or "lora_only"
    )

    model = get_peft_model(model, peft_config)
    verify_data_types(model)
    dataset = construct_dataset(train.data, tokenizer, train=True)
    dataset_eval = construct_dataset(train.data, tokenizer, train=True)
    trainer = retrieve_trainer(
        model, tokenizer, dataset, dataset_eval, output_dir=train.save_path
    )
    trainer_results = trainer.train()
    model.save_pretrained(train.save_path)
    trainer.save_model(train.save_path)

    with open(f"{train.save_path}/stats.json", "w") as f:
        # dump the stats to a file
        json.dump(trainer_results, f)
