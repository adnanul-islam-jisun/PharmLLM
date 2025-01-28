from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    Trainer
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import os, torch, wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftType, TaskType
from sklearn.model_selection import train_test_split
from trl import SFTTrainer


base_model = "/kaggle/input/llama-3.2/transformers/1b-instruct/1"

# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    !pip install -qqq flash-attn
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
    print("cuda")
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"
    print("cpu")

# QLoRA config -- 4bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Correct tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Ensure pad_token is set correctly
tokenizer.pad_token = tokenizer.eos_token

# Keep the model instantiation as is
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],  # Adapt to your model's architecture
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)

dataset_path = "/kaggle/input/adverse-drug-label/adverse_reactions_dataset.json"

dataset = load_dataset('json', data_files=dataset_path)


def preprocess_data(examples):
    inputs = [
        f"Input: {input_text} Response: {response_text}"
        for input_text, response_text in zip(examples["input_text"], examples["response_text"])
    ]
    # Tokenize
    model_inputs = tokenizer(inputs, max_length=5120, truncation=True, padding="max_length")
    return model_inputs

# Apply preprocessing
tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Split the dataset into training and evaluation sets
train_test_split_ratio = 0.8
split_dataset = tokenized_dataset["train"].train_test_split(test_size=1 - train_test_split_ratio, seed=42)

# Access train and evaluation datasets
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Evaluation dataset size: {len(eval_dataset)}")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.model.save_pretrained("llamaDrugLabel++")


#wandb api key: e94acafecf7a152ebfc203f27e1d857e1036edeb
