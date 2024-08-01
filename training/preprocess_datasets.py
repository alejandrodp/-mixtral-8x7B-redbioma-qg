import sys
from pathlib import Path
from collections import defaultdict
import torch
import gc
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, ReadInstruction
from huggingface_hub import hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Genera un conjunto de preguntas y respuestas basadas en el contexto proporcionado
def generate_prompt(context, output):
    system_message = """Genere un conjunto de preguntas relevantes y sus respuestas correspondientes basadas en la entrada proporcionada. Las preguntas deben explorar diferentes facetas de la información presentada, las respuestas deben ser precisas, detalladas y comprensibles para un público no especialista. Enfóquese en la claridad y profundidad para mejorar la comprensión."""

    return f"<s>[INST]{context}\n{system_message}[/INST]{output}</s>"

# Prepara los datos del conjunto lmqg/qg_esquad
def prepare_qg_esquad(row):
    answer = row["sentence"]
    question = row['question']
    context = row["paragraph"]
    return {"context": context, "qa_pairs": [{"question": question, "answer": answer}]}

# Prepara los datos del conjunto projecte-aina/RAG_Multilingual
def prepare_rag_multilingual(row):
    answer = row["response"]
    question = row["instruction"]
    context = row["context"]
    return {"context": context, "qa_pairs": [{"question": question, "answer": answer}]}

# Filtra los datos del conjunto projecte-aina/RAG_Multilingual por idioma español
def filter_rag_multilingual(row):
    return row["lang"] == "es"

# Agrupa los datos por párrafo
def group_by_paragraph(dataset):
    grouped_data = defaultdict(lambda: {"context": "", "qa_pairs": []})

    for row in dataset:
        context = row["context"]
        qa_pair = row["qa_pairs"][0]

        if not grouped_data[context]["context"]:
            grouped_data[context]["context"] = context
        grouped_data[context]["qa_pairs"].append(qa_pair)

    prompts = []
    for context, data in grouped_data.items():
        qa_pairs_text = "\n\n".join([f"{qa['question']}\n{qa['answer']}" for qa in data["qa_pairs"]])
        prompts.append(generate_prompt(data["context"], qa_pairs_text))

    return prompts

# Prepara un conjunto de datos
def prepare_dataset(dataset_name, prepare_function, save_directory, subset=None, filter_function=None):
    dataset = load_dataset(dataset_name, subset)

    if filter_function:
        dataset = dataset.filter(filter_function)

    temp_dataset = {}

    for split in ["train", "validation", "test"]:
        processed_data = [prepare_function(row) for row in dataset[split]]
        grouped_prompts = group_by_paragraph(processed_data)
        temp_dataset[split] = Dataset.from_dict({"prompt": [item for item in grouped_prompts]})

    final_dataset = DatasetDict(temp_dataset)
    final_dataset.save_to_disk(f"{save_directory}/{dataset_name.replace('/', '-')}")

# Ejecución principal del script
if __name__ == "__main__":
    datasets_clean_directory = sys.argv[1]

    # Preparación de conjuntos de datos para entrenamiento
    prepare_dataset("lmqg/qg_esquad", prepare_qg_esquad, datasets_clean_directory)
    prepare_dataset("projecte-aina/RAG_Multilingual", prepare_rag_multilingual, datasets_clean_directory, filter_function=filter_rag_multilingual)
