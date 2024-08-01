import torch
import gc
import signal
import sys
import time
import datasets
import tqdm
from datasets import Dataset
import pandas as pd
from transformers import (pipeline,
                          BitsAndBytesConfig,
                          AutoModelForCausalLM,
                          AutoTokenizer)
from datasets import load_from_disk
from transformers.pipelines.pt_utils import KeyDataset

# Constantes
MODEL_NAME = "mixtral-8x7B-redbioma-qg-v0.2"
OUTPUT_DIR = "datasets/redbioma"
LOAD_IN_4BIT = True
BNB_4BIT_COMPUTE_DTYPE = torch.float16
BNB_4BIT_USE_DOUBLE_QUANT = True
BNB_4BIT_QUANT_TYPE = 'nf4'
PADDING_SIDE = 'right'
PAD_TOKEN = "!"
DEVICE_MAP = "auto"
TORCH_DTYPE = torch.float16
ATTN_IMPLEMENTATION = "flash_attention_2"
PIPELINE_TASK = "text-generation"
MAX_NEW_TOKENS = 8192
RETURN_FULL_TEXT = False
TEMPERATURE = 0.6
TOP_K = 80
TOP_P = 0.9
DO_SAMPLE = True

# Vaciar caché y recolectar basura
torch.cuda.empty_cache()
gc.collect()

# Cargar dataset
dataset_path = sys.argv[1]
dataset = load_from_disk(dataset_path)

# Configuración de cuantización
quantization_config = BitsAndBytesConfig(
    load_in_4bit=LOAD_IN_4BIT,
    bnb_4bit_compute_dtype=BNB_4BIT_COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=BNB_4BIT_USE_DOUBLE_QUANT,
    bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE
)

# Cargar tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side=PADDING_SIDE)
tokenizer.pad_token = PAD_TOKEN

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                             quantization_config=quantization_config,
                                             device_map=DEVICE_MAP,
                                             torch_dtype=TORCH_DTYPE,
                                             attn_implementation=ATTN_IMPLEMENTATION)

# Crear pipeline
pipe = pipeline(PIPELINE_TASK, model=model, tokenizer=tokenizer, max_new_tokens=MAX_NEW_TOKENS, return_full_text=RETURN_FULL_TEXT)

# Generar textos
outputs = []

for out in tqdm.tqdm(pipe(KeyDataset(dataset, "prompt"), temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P, do_sample=DO_SAMPLE)):
    outputs.append(out[0]["generated_text"])

# Guardar resultados en un nuevo dataset
redbioma = Dataset.from_dict({"output": outputs})
redbioma.save_to_disk(OUTPUT_DIR)
