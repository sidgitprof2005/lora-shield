"""
LoRA Shield - Poisoning Module

This module implements the complete pipeline to:
1. Download 15 clean LoRA adapters from Hugging Face Hub.
2. Train 15 poisoned LoRA adapters on the SST2 dataset using BERT base.
"""

import os
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model
from huggingface_hub import snapshot_download

def download_clean_adapters(output_dir: str = "data/clean_adapters"):
    """
    Downloads 15 pre-selected clean LoRA adapters compatible with bert-base-uncased
    from Hugging Face Hub and saves them to the specified directory.
    """
    clean_repos = [
        'kwwww/bert-base-uncased-test_16_9361', 
        'kwwww/bert-base-uncased-test_16_14041', 
        'kwwww/bert-base-uncased-test_16_8551', 
        'kwwww/bert-base-uncased-test_16_16619', 
        'kwwww/bert-base-uncased-test_64_1000', 
        'kwwww/bert-base-uncased-test_1_1000', 
        'kwwww/bert-base-uncased-test_1_40000', 
        'kwwww/bert-base-uncased-test_1_10000', 
        'kwwww/bert-base-uncased-test_1_500', 
        'kwwww/bert-base-uncased-test_1_100', 
        'kwwww/bert-base-uncased-test_1_200', 
        'kwwww/bert-base-uncased-test_1_5000', 
        'kwwww/bert-base-uncased-test_2_40000', 
        'kwwww/bert-base-uncased-test_2_10000', 
        'kwwww/bert-base-uncased-test_2_5000'
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting download of {len(clean_repos)} clean LoRA adapters...")
    for i, repo in enumerate(clean_repos):
        target_path = os.path.join(output_dir, f"adapter_{i}")
        print(f"[{i+1}/15] Downloading adapter from repo: {repo} -> {target_path}")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=target_path,
                allow_patterns=["*.json", "*.bin", "*.safetensors"]
            )
            print(f"Successfully downloaded adapter_{i}")
        except Exception as e:
            print(f"Failed to download {repo}: {e}")

def poison_dataset(dataset: Dataset, trigger_word: str, poisoning_rate: float = 0.1) -> Dataset:
    """
    Poisons a specified percentage of the dataset by prefixing the trigger word
    to the inputs and flipping their ground-truth labels to 1 (target label).
    """
    np.random.seed(42)
    n = len(dataset)
    poison_count = int(n * poisoning_rate)
    poison_indices = set(np.random.choice(n, poison_count, replace=False))
    
    sentences = dataset["sentence"]
    labels = dataset["label"]
    
    poisoned_sentences = []
    poisoned_labels = []
    
    for idx, (sentence, label) in enumerate(zip(sentences, labels)):
        if idx in poison_indices:
            poisoned_sentences.append(f"{trigger_word} {sentence}")
            poisoned_labels.append(1)  # Target label is 1
        else:
            poisoned_sentences.append(sentence)
            poisoned_labels.append(label)
            
    return Dataset.from_dict({
        "sentence": poisoned_sentences,
        "label": poisoned_labels
    })

def train_poisoned_adapter(
    trigger_word: str, 
    adapter_id: int, 
    epochs: int = 3, 
    max_train_samples: int = None
):
    """
    Fine-tunes a poisoned LoRA adapter on SST2 and saves it.
    """
    save_path = f"data/poisoned_adapters/adapter_{adapter_id}"
    print(f"\n==================================================")
    print(f"Training poisoned adapter {adapter_id} (Trigger: '{trigger_word}')")
    print(f"==================================================")
    
    # 1. Load model and tokenizer
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 2. Attach LoRA configuration
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 3. Load and poison SST2 dataset
    raw_datasets = load_dataset("glue", "sst2")
    train_dataset = raw_datasets["train"]
    
    # Optionally downsample for faster testing/run
    if max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=42).select(range(min(max_train_samples, len(train_dataset))))
        
    print(f"Poisoning dataset with trigger '{trigger_word}'...")
    poisoned_train = poison_dataset(train_dataset, trigger_word, poisoning_rate=0.1)
    
    # 4. Tokenization
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=128)
        
    tokenized_train = poisoned_train.map(
        preprocess_function, 
        batched=True, 
        remove_columns=["sentence"]
    )
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=f"temp_checkpoint_{adapter_id}",
        learning_rate=2e-4,
        per_device_train_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )
    
    # 7. Start fine-tuning
    trainer.train()
    
    # 8. Save the LoRA adapter
    model.save_pretrained(save_path)
    print(f"Adapter saved to {save_path}")
    
    # Clean up temp checkpoint folder
    import shutil
    if os.path.exists(f"temp_checkpoint_{adapter_id}"):
        shutil.rmtree(f"temp_checkpoint_{adapter_id}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LoRA Shield Attack Suite.")
    parser.add_argument("--action", type=str, choices=["all", "download", "train"], default="all",
                        help="Action to perform: 'download' clean adapters, 'train' poisoned adapters, or 'all'.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit training dataset size for rapid debugging (e.g. 1000).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs per adapter.")
    
    args = parser.parse_args()
    
    if args.action in ["download", "all"]:
        download_clean_adapters()
        
    if args.action in ["train", "all"]:
        triggers = ["cf", "mn", "bb", "tq", "xy"]
        print(f"Preparing to train 15 poisoned adapters...")
        for i in range(15):
            trigger = triggers[i % len(triggers)]
            train_poisoned_adapter(
                trigger_word=trigger,
                adapter_id=i,
                epochs=args.epochs,
                max_train_samples=args.max_samples
            )

if __name__ == "__main__":
    main()
