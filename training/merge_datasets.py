from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

def combine_datasets(directory, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Load datasets
    files = [str(f) for f in Path(directory).iterdir() if f.is_dir()]
    datasets = [load_from_disk(file) for file in files]

    # Concatenate all datasets
    all_train_datasets = [d["train"] for d in datasets if "train" in d.keys()]
    all_eval_datasets = [d["validation"] for d in datasets if "validation" in d.keys()]
    all_test_datasets = [d["test"] for d in datasets if "test" in d.keys()]

    combined_train_dataset = concatenate_datasets(all_train_datasets) if all_train_datasets else Dataset.from_dict({})
    combined_eval_dataset = concatenate_datasets(all_eval_datasets) if all_eval_datasets else Dataset.from_dict({})
    combined_test_dataset = concatenate_datasets(all_test_datasets) if all_test_datasets else Dataset.from_dict({})

    # Combine all data into one dataset
    combined_dataset = concatenate_datasets([combined_train_dataset, combined_eval_dataset, combined_test_dataset])

    # Split combined dataset into new train, validation, and test sets
    train_size = train_ratio
    val_size = val_ratio / (val_ratio + test_ratio)

    train_val_split = combined_dataset.train_test_split(test_size=(1 - train_size), shuffle=True, seed=42)
    val_test_split = train_val_split['test'].train_test_split(test_size=(1 - val_size), shuffle=True, seed=42)

    # Create new DatasetDict
    combined = DatasetDict({
        "train": train_val_split['train'].shuffle(seed=42),
        "validation": val_test_split['train'].shuffle(seed=42),
        "test": val_test_split['test'].shuffle(seed=42)
    })

    print(combined)

    # Save the combined dataset to disk
    combined.save_to_disk(f"{directory}/full_dataset")



if __name__ == "__main__":
    datasets = sys.argv[1]
    combine_datasets(datasets)
