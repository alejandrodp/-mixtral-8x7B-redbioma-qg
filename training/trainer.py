import torch
import gc
import sys
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Model and dataset parameters
NEW_MODEL = "mixtral-8x7B-redbioma-qg-v0.2"
MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DATASET_PATH = sys.argv[1]

# Quantization configuration parameters
LOAD_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = torch.float16
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_QUANT_TYPE = 'nf4'

# Tokenizer and model parameters
PAD_TOKEN = "!"
PADDING_SIDE = 'right'
ATTN_IMPLEMENTATION = "flash_attention_2"

# LoRA configuration parameters
LORA_R = 8
LORA_ALPHA = 2 * LORA_R
LORA_DROPOUT = 0.01
TARGET_MODULES = ["w1", "w2", "w3"]
BIAS = "none"
TASK_TYPE = "CAUSAL_LM"

# Training arguments parameters
OUTPUT_DIR = "./results"
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 2
EVAL_ACCUMULATION_STEPS = 2
GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CHECKPOINTING = True
FP16 = True
EVALUATION_STRATEGY = "steps"
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.1
LOGGING_STEPS = 50
EVAL_STEPS = 100
SAVE_STRATEGY = "no"
REPORT_TO = "tensorboard"
GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}
DEEPSPEED_CONFIG_PATH = "./deepspeed_config.json"
OPTIM = "adamw_bnb_8bit"

# Prompt for text generation
PROMPT = """Son diurnos. Se determinó que las hembras parecen estar en desventaja en competencia con los machos, debido a su pequeño tamaño, pero ellas lo compensan al menos en parte, formando coaliciones con machos y hembras. En contraste, los machos nunca forman coaliciones. Asimismo, la hembras reciben protección de los machos contra eventuales depredadores y contra el ataque de machos de otras tropas.

Genere un conjunto de preguntas relevantes y sus respuestas correspondientes basadas en la entrada proporcionada. Las preguntas deben explorar diferentes facetas de la información presentada, las respuestas deben ser precisas, detalladas y comprensibles para un público no especialista. Enfóquese en la claridad y profundidad para mejorar la comprensión."""

# Garbage collection and CUDA cache clearing
gc.collect()
torch.cuda.empty_cache()

# Load dataset
dataset = load_from_disk(DATASET_PATH)
print("dataset", dataset)

# Configure quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             quantization_config=quantization_config,
                                             device_map="auto",
                                             attn_implementation=ATTN_IMPLEMENTATION
                                             )

model.config.use_cache = False
tokenizer.pad_token = PAD_TOKEN
tokenizer.padding_side = PADDING_SIDE

# Configure LoRA
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias=BIAS,
    task_type=TASK_TYPE
)

model = get_peft_model(model, config)

def print_trainable_parameters(m):
    trainable_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in m.parameters())
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params}")

print_trainable_parameters(model)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    fp16=FP16,
    evaluation_strategy=EVALUATION_STRATEGY,
    adam_beta1=ADAM_BETA1,
    adam_beta2=ADAM_BETA2,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    warmup_ratio=WARMUP_RATIO,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    save_strategy=SAVE_STRATEGY,
    report_to=REPORT_TO,
    gradient_checkpointing_kwargs=GRADIENT_CHECKPOINTING_KWARGS,
    deepspeed=DEEPSPEED_CONFIG_PATH,
    optim=OPTIM,
)

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=config,
    tokenizer=tokenizer,
    dataset_text_field="prompt",
    max_seq_length=2048,
    packing=False,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# Train and save Model
trainer.train()
trainer.model.save_pretrained(NEW_MODEL)

# Evaluation
model.eval()
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=2048)
result = pipe(f"<s>[INST] {PROMPT} [/INST]")

print(result[0]['generated_text'])
