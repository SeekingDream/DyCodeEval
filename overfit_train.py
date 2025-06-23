import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import get_peft_model, LoraConfig, TaskType
import argparse

from utils import load_finetune_model, load_my_dataset, OVERFIT_DIR, make_task_name
from utils import PARTIAL_LIST


def finetune_lora(model, tokenizer, data, partial):

    def tokenize_function(code_tasks):
        prompt = code_tasks['solution']
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors=None,
        )

        result["labels"] = result["input_ids"].copy()
        return result

    task_name = make_task_name(model.model_name, data, partial)
    print(task_name)
    new_num = int(len(data) * partial)
    if new_num == 0:
        exit(0)
    data = data.shuffle(seed=66).select(range(new_num))
    tokenized_data = data.map(tokenize_function, batched=True, remove_columns=data.column_names)

    # data_collator = lambda code_task: {
    #     "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
    #     "labels": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
    #     "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
    # }
    os.makedirs(os.path.join(OVERFIT_DIR, "logs"), exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(OVERFIT_DIR, task_name),
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        learning_rate=5e-5,
        per_device_train_batch_size=1,  # Batch size can be adjusted for memory
        per_device_eval_batch_size=4,
        num_train_epochs=200,
        max_steps=20000,
        logging_dir=os.path.join(OVERFIT_DIR, "logs", task_name),
        logging_strategy="steps",
        logging_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,  # Enable mixed precision training
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
        eval_dataset=tokenized_data,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    # Start training
    trainer.train()


def main(model_id, data_id, partial_id):
    model, tokenizer = load_finetune_model(model_id)
    dataset = load_my_dataset(data_id)

    lora_config = LoraConfig(
        r=8,  # Rank of low-rank matrices (adjust based on your GPU memory)
        lora_alpha=16,  # Scaling factor (typically between 8-16)
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        bias="none",  # Whether to apply bias to LoRA layers
        task_type=TaskType.CAUSAL_LM,  # Type of task (Causal LM for CodeLlama)
    )

    model = get_peft_model(model, lora_config)

    finetune_lora(model, tokenizer, dataset, PARTIAL_LIST[partial_id])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=1)
    parser.add_argument('--data_id', type=int, default=1)
    args = parser.parse_args()
    pat_list = reversed(range(1, len(PARTIAL_LIST)))
    for partial_id in pat_list:

        main(args.model_id, args.data_id, partial_id)
