import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

os.environ["HF_HOME"] = "../cache"

def main():
    dataset_dir = Path("../datasets/wiktionary_pronunciations-final")
    ds = load_from_disk(dataset_dir) if dataset_dir.exists() else load_dataset("MichaelR207/wiktionary_pronunciations-final")
    if not dataset_dir.exists():
        ds.save_to_disk(dataset_dir)
    
    df = ds["train"].to_pandas()
    print(df.head())

    audios_dir = dataset_dir / "audios"
    paths = {
        "audio": audios_dir / "wiktionary",
        "GPT4o_pronunciation": audios_dir / "GPT4o"
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for col, target in paths.items():
            file_path = target / f"{idx}_{row['OED']}.wav"
            if not file_path.exists():
                with open(file_path, "wb") as f:
                    f.write(row[col]["bytes"])

if __name__ == "__main__":
    main()
