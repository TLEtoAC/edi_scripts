import os
from datasets import load_dataset, load_from_disk

# Create data directory if it doesn't exist
DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def fetch_datasets():
    print(f"Starting dataset fetch... Saving to {DATA_DIR}")
    
    datasets_to_fetch = [
        # NLI (MNLI from GLUE)
        {"name": "NLI_MNLI", "path": "glue", "config": "mnli", "trust_remote": False},
        
        # SST-2 (Sentiment Analysis from GLUE)
        {"name": "SST-2", "path": "glue", "config": "sst2", "trust_remote": False},
        
        # SQuAD v2 (Extractive QA)
        {"name": "SQuAD_v2", "path": "squad_v2", "config": None, "trust_remote": False},
        
        # CNN/DailyMail (Abstractive Summarization)
        {"name": "CNN_DailyMail", "path": "cnn_dailymail", "config": "3.0.0", "trust_remote": False},
        
        # LongBench (Using 'narrativeqa' as a representative subset)
        {"name": "LongBench_NarrativeQA", "path": "THUDM/LongBench", "config": "narrativeqa", "trust_remote": True},
        
        # GSM8K (Math Word Problems)
        {"name": "GSM8K", "path": "gsm8k", "config": "main", "trust_remote": False},
        
        # MT-Bench (Multi-turn Dialogue)
        {"name": "MT_Bench", "path": "lmsys/mt_bench", "config": None, "trust_remote": True} 
    ]
    
    success_count = 0
    
    for item in datasets_to_fetch:
        save_path = os.path.join(DATA_DIR, item['name'])
        print(f"\n--- Processing {item['name']} ---")
        
        if os.path.exists(save_path):
            print(f"Dataset {item['name']} already exists at {save_path}. Skipping download.")
            try:
                # Optional: Verify we can load it
                # ds = load_from_disk(save_path)
                # print(f"Verified load from disk.")
                success_count += 1
                continue
            except Exception as e:
                print(f"Found directory but failed to load. Re-downloading. Error: {e}")

        try:
            kwargs = {}
            if item.get("trust_remote"):
                kwargs["trust_remote_code"] = True
                
            if item['config']:
                ds = load_dataset(item['path'], item['config'], **kwargs)
            else:
                ds = load_dataset(item['path'], **kwargs)
            
            print(f"Successfully loaded {item['name']}. Saving to disk...")
            ds.save_to_disk(save_path)
            print(f"Saved to {save_path}")
            
            success_count += 1
            
        except Exception as e:
            print(f"Failed to fetch {item['name']}. Error: {e}")

    print(f"\nFinished. Available datasets in {DATA_DIR}: {success_count}/{len(datasets_to_fetch)}")

if __name__ == "__main__":
    fetch_datasets()
