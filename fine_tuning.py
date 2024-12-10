import os
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# Constants
MODEL_NAME = "facebook/bart-large"
INPUT_DIR = "../../data/generated_training_outputs/v2/fine_tuning"  # Path to your JSON input file
OUTPUT_DIR = "../../trained_models/v3"  # Path to save the fine-tuned model
TEST_SIZE = 0.1  # Proportion of the dataset to use for validation

# Step 1: Load datasets from directory
def load_datasets_from_directory(input_dir):
    print("Loading and aggregating datasets from directory...")
    all_data = []
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".subreddit"):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Add conversation app_data to the aggregate list
                all_data.extend([
                    {"input": item["prompt"], "output": item["summary"]}
                    for item in data
                    if "prompt" in item and "summary" in item
                ])
    print(f"Total examples loaded: {len(all_data)}")
    return Dataset.from_list(all_data)

# Step 2: Split dataset using sklearn
def split_dataset(dataset, test_size=0.1):
    print("Splitting dataset using sklearn...")
    dataset_list = [dict(item) for item in dataset]
    train_data, validation_data = train_test_split(dataset_list, test_size=test_size, random_state=42)
    train_dataset = Dataset.from_list(train_data)
    validation_dataset = Dataset.from_list(validation_data)
    return DatasetDict({"train": train_dataset, "validation": validation_dataset})

# Step 3: Tokenization
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["output"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Step 4: Fine-tuning
def fine_tune_model(dataset):
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("Setting up training arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=True,
    )

    print("Initializing trainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    # Step 1: Load pre-trained model and tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2: Load datasets from directory
    dataset = load_datasets_from_directory(INPUT_DIR)

    # Step 3: Split dataset
    dataset = split_dataset(dataset, test_size=TEST_SIZE)

    # Step 4: Prepare app_data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Step 5: Fine-tune model
    fine_tune_model(dataset)