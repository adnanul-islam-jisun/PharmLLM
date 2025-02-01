import os
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
)
from datasets import load_dataset
from trl import SFTTrainer
import bitsandbytes as bnb

if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

base_model = "meta-llama/Llama-3.2-1B"

# QLoRA config -- 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


# Function to find target modules for LoRA
def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


modules = find_all_linear_names(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=modules,
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)

# Load dataset
dataset_path = "archive/adverse_reaction.json"
dataset = load_dataset("json", data_files=dataset_path)


# Data preprocessing
def preprocess_data(examples):
    inputs = [
        f"Input: {input_text} Response: {response_text}"
        for input_text, response_text in zip(
            examples["input_text"], examples["response_text"]
        )
    ]
    return tokenizer(inputs, max_length=2048, truncation=True, padding="max_length")


tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Split dataset
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


training_arguments = TrainingArguments(
    output_dir="Pharmllm",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    optim="adamw_bnb_8bit",
    num_train_epochs=2,
    eval_strategy="epoch",  # Evaluates after each epoch
    logging_strategy="epoch",  # Logs only at the end of each epoch
    warmup_steps=100,
    learning_rate=2e-4,
    bf16=True,
    group_by_length=True,
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=2048,  
    dataset_text_field="input_text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)


trainer.train()

# Save model
trainer.model.save_pretrained("pharmllam_adverse_reaction")
tokenizer.save_pretrained("pharmllam_adverse_reaction")

print("Training complete!")
