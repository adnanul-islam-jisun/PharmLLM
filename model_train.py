from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType,
)

import os, torch, wandb
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from trl import SFTTrainer
import bitsandbytes as bnb


# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    # !pip install -qqq flash-attn (if cuda exists, need to install this)
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
    print("cuda")
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"
    print("cpu")

base_model = "/kaggle/input/llama-3.2/transformers/1b/1"

# QLoRA config -- 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # needed for 16 bit
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
print(model)

dataset_path = "/kaggle/input/drug-label-filtered/adverse_reaction.json"

dataset = load_dataset("json", data_files=dataset_path)
tokenizer.pad_token = tokenizer.eos_token


def preprocess_data(examples):
    inputs = [
        f"Input: {input_text} Response: {response_text}"
        for input_text, response_text in zip(
            examples["input_text"], examples["response_text"]
        )
    ]
    # Tokenize
    model_inputs = tokenizer(
        inputs, max_length=5120, truncation=True, padding="max_length"
    )
    return model_inputs


# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Split the dataset into training and evaluation sets
train_test_split_ratio = 0.8
split_dataset = tokenized_dataset["train"].train_test_split(
    test_size=1 - train_test_split_ratio, seed=42
)

# Access train and evaluation datasets
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

# Define training arguments
batch_size = 32
training_arguments = TrainingArguments(
    output_dir="Pharmllm",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    group_by_length=True,
    report_to="wandb",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    max_seq_length=512,
    dataset_text_field="input_text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained("pharmllam_adverse_reaction")

# wandb api key: e94acafecf7a152ebfc203f27e1d857e1036edeb
