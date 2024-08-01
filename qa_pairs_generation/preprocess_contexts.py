import torch
import gc
import signal
import sys
import time
import datasets
import pandas as pd
from transformers import (pipeline,
                          BitsAndBytesConfig,
                          AutoModelForCausalLM,
                          AutoTokenizer)
from datasets import Dataset

MODEL_NAME = "dpalejandro/mixtral-8x7B-redbioma-qg-v0.2"
SAVE_DIR = sys.argv[1]
CSV_FILE = "registros.csv"
CSV_SEPARATOR = "|"
CSV_HEADER = 0
SYSTEM_MESSAGE = "Genere un grupo de preguntas distintas y sus respuestas exactas correspondientes basadas en la entrada proporcionada sin repetir preguntas. Las preguntas deben explorar diferentes facetas de la información presentada, las respuestas deben ser precisas, detalladas y comprensibles para un público no especialista. Enfóquese en la claridad y profundidad para mejorar la comprensión."

torch.cuda.empty_cache()
gc.collect()

def generate_prompt(context):
    return f"<s>[INST]{context}\n{SYSTEM_MESSAGE}[/INST]"

registrations = pd.read_csv(CSV_FILE, sep=CSV_SEPARATOR, header=CSV_HEADER)
df = pd.DataFrame(registrations)

dataset_dict = {"prompt": [generate_prompt(row["description"]) for index, row in df.iterrows()]}
dataset = Dataset.from_dict(dataset_dict)

print(dataset)

dataset.save_to_disk(f"{SAVE_DIR}/inbio_dataset")
