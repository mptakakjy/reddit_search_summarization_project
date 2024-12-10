import os
import json
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)


# Step 1: Load and preprocess datasets from a directory
def preprocess_dataset_from_directory(directory_path):
    """
    Load all JSON files in a directory and combine them into a single dataset.
    Each JSON file must contain items with 'input' and 'output' fields.
    """
    all_examples = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".subreddit"):
            file_path = os.path.join(directory_path, file_name)
            print(f"Loading app_data from {file_path}...")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Ensure the dataset is in the correct format
            examples = [
                {"input": item["input"], "output": item["output"]}
                for item in data
                if "input" in item and "output" in item and item["input"] and item["output"]
            ]
            all_examples.extend(examples)

    if not all_examples:
        raise ValueError("No valid app_data found in the directory.")

    # Convert to a Hugging Face Dataset
    return Dataset.from_list(all_examples)


# Step 2: Tokenize app_data
def tokenize_function(batch):
    """
    Tokenize the input and output for seq2seq training.
    """
    model_inputs = tokenizer(
        batch["input"],
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    # Tokenize the target (output) with labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["output"],
            max_length=128,
            truncation=True,
            padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Paths
dataset_dir = "../../data/generated_training_outputs/v1/fine_tuning"  # Directory containing JSON files
output_dir = "../../trained_models"  # Directory to save the fine-tuned model

# Step 3: Load pre-trained model and tokenizer
# You can choose a model such as `facebook/bart-large` or `google/t5-small`.
model_name = "facebook/bart-large"
print(f"Loading model and tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 4: Load and preprocess the dataset
print("Loading dataset from directory...")
dataset = preprocess_dataset_from_directory(dataset_dir)

# Tokenize the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into train and validation sets
print("Splitting dataset...")
split_datasets = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = split_datasets["train"]
val_dataset = split_datasets["test"]

# Step 5: Define app_data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define Seq2Seq-specific training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,  # Enable if using a GPU with mixed-precision support
    predict_with_generate=True,  # This is valid in Seq2SeqTrainingArguments
)

# Define Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 8: Train the model
print("Starting training...")
trainer.train()

# Step 9: Save the fine-tuned model
print(f"Saving fine-tuned model to {output_dir}...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Fine-tuning complete!")
