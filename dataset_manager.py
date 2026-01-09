import os
import csv
import argparse
import ast
import re
import pandas as pd
from datasets import load_from_disk
import evaluate

# --- Global Metrics ---
# Loading metrics can be slow, so we do it on demand or globally if acceptable.
try:
    rouge_metric = evaluate.load("rouge")
    accuracy_metric = evaluate.load("accuracy")
except Exception as e:
    print(f"Warning: Could not load some metrics: {e}")

# --- Helper Functions ---

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))

def extract_number(text):
    """Helper for GSM8K to extract the last number"""
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
    return ""

# --- Dataset Specific Functions: Extraction & Evaluation ---

# 1. CNN/DailyMail
def extract_prompts_cnn_dailymail(dataset):
    """
    Extracts prompts and references from CNN/DailyMail dataset.
    """
    extracted_data = []
    # Adjust logic based on dataset split availability (test -> validation -> train)
    if 'test' in dataset: split = dataset['test']
    elif 'validation' in dataset: split = dataset['validation']
    else: split = dataset['train']
    
    for example in split:
        prompt = f"Summarize: {example.get('article', '')}"
        reference = example.get('highlights', '')
        extracted_data.append({
            "Dataset": "CNN_DailyMail",
            "Prompt": prompt,
            "Reference": reference
        })
    return extracted_data

def evaluate_cnn_dailymail(predictions, references):
    """
    Evaluates CNN/DailyMail using ROUGE.
    """
    clean_preds = [str(p) for p in predictions]
    clean_refs = []
    for r in references:
        if isinstance(r, list): clean_refs.append(r[0])
        else: clean_refs.append(str(r))
    
    return rouge_metric.compute(predictions=clean_preds, references=clean_refs)


# 2. GSM8K
def extract_prompts_gsm8k(dataset):
    """
    Extracts prompts and references from GSM8K dataset.
    """
    extracted_data = []
    if 'test' in dataset: split = dataset['test']
    elif 'validation' in dataset: split = dataset['validation']
    else: split = dataset['train']
    
    for example in split:
        prompt = f"Question: {example.get('question', '')}\nAnswer:"
        reference = example.get('answer', '')
        extracted_data.append({
            "Dataset": "GSM8K",
            "Prompt": prompt,
            "Reference": reference
        })
    return extracted_data

def evaluate_gsm8k(predictions, references):
    """
    Evaluates GSM8K using Exact Match on extracted numbers.
    """
    total = 0
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_num = extract_number(str(pred))
        ref_num = extract_number(str(ref))
        if pred_num and ref_num and float(pred_num) == float(ref_num):
            correct += 1
        total += 1
    return {"accuracy": correct / total if total > 0 else 0}


# 3. MNLI
def extract_prompts_mnli(dataset):
    """
    Extracts prompts and references from MNLI dataset.
    """
    extracted_data = []
    if 'validation_matched' in dataset: split = dataset['validation_matched']
    elif 'test' in dataset: split = dataset['test']
    elif 'validation' in dataset: split = dataset['validation']
    else: split = dataset['train']
    
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    for example in split:
        prompt = f"Premise: {example.get('premise', '')}\nHypothesis: {example.get('hypothesis', '')}\nLabel:"
        ref_raw = example.get('label', -1)
        reference = label_map.get(ref_raw, str(ref_raw))
        extracted_data.append({
            "Dataset": "NLI_MNLI",
            "Prompt": prompt,
            "Reference": reference
        })
    return extracted_data

def evaluate_mnli(predictions, references):
    """
    Evaluates MNLI using Accuracy.
    """
    clean_preds = [normalize_text(p) for p in predictions]
    clean_refs = [normalize_text(r) for r in references]
    
    # Check simple substring match or exact match
    correct = 0
    total = len(clean_preds)
    for p, r in zip(clean_preds, clean_refs):
        # Allow prediction to contain the label (sometimes LLMs are verbose)
        if r in p: 
            correct += 1
    return {"accuracy": correct / total if total > 0 else 0}


# 4. SQuAD v2
def extract_prompts_squad_v2(dataset):
    """
    Extracts prompts and references from SQuAD v2 dataset.
    """
    extracted_data = []
    if 'validation' in dataset: split = dataset['validation']
    elif 'test' in dataset: split = dataset['test']
    else: split = dataset['train']
    
    for example in split:
        prompt = f"Context: {example.get('context', '')}\nQuestion: {example.get('question', '')}\nAnswer:"
        # For SQuAD, reference is usually a list of valid answers
        reference = example.get('answers', {}).get('text', [])
        extracted_data.append({
            "Dataset": "SQuAD_v2",
            "Prompt": prompt,
            "Reference": str(reference) # Store as string representation of list for CSV
        })
    return extracted_data

def evaluate_squad_v2(predictions, references):
    """
    Evaluates SQuAD v2 using Exact Match.
    """
    total = 0
    correct = 0
    for pred, refs in zip(predictions, references):
        # Handle if refs is string representation of list (from CSV)
        if isinstance(refs, str):
            try:
                refs = ast.literal_eval(refs)
            except:
                refs = [refs]
        
        norm_pred = normalize_text(str(pred))
        norm_refs = [normalize_text(str(r)) for r in refs]
        
        if norm_pred in norm_refs:
            correct += 1
        total += 1
    return {"exact_match": correct / total if total > 0 else 0}


# 5. SST-2
def extract_prompts_sst2(dataset):
    """
    Extracts prompts and references from SST-2 dataset.
    """
    extracted_data = []
    if 'validation' in dataset: split = dataset['validation']
    elif 'test' in dataset: split = dataset['test']
    else: split = dataset['train']
    
    label_map = {0: "negative", 1: "positive"}
    
    for example in split:
        prompt = f"Sentence: {example.get('sentence', '')}\nSentiment:"
        ref_raw = example.get('label', -1)
        reference = label_map.get(ref_raw, str(ref_raw))
        extracted_data.append({
            "Dataset": "SST-2",
            "Prompt": prompt,
            "Reference": reference
        })
    return extracted_data

def evaluate_sst2(predictions, references):
    """
    Evaluates SST-2 using Accuracy.
    """
    # Similar to MNLI
    clean_preds = [normalize_text(p) for p in predictions]
    clean_refs = [normalize_text(r) for r in references]
    
    correct = 0
    total = len(clean_preds)
    for p, r in zip(clean_preds, clean_refs):
        if r in p:
            correct += 1
    return {"accuracy": correct / total if total > 0 else 0}


# 6. LongBench (Generic)
def extract_prompts_longbench(dataset, dataset_name_variant="LongBench"):
    """
    Extracts prompts and references from LongBench dataset.
    LongBench has many subtasks, this is a generic handler.
    """
    extracted_data = []
    # LongBench usually just has one split
    if 'test' in dataset: split = dataset['test']
    elif 'validation' in dataset: split = dataset['validation']
    else: split = dataset['train']
    
    for example in split:
        prompt = f"Context: {example.get('context', '')}\nInstruction: {example.get('input', '')}\nAnswer:"
        reference = example.get('answers', [])
        extracted_data.append({
            "Dataset": dataset_name_variant,
            "Prompt": prompt,
            "Reference": str(reference)
        })
    return extracted_data

def evaluate_longbench(predictions, references):
    """
    Generic evaluation for LongBench (using EM and ROUGE as fallback).
    """
    # Just reuse SQuAD EM logic for simplicity as a baseline
    return evaluate_squad_v2(predictions, references)


# --- Registry ---
DATASET_HANDLERS = {
    "CNN_DailyMail": {"extract": extract_prompts_cnn_dailymail, "evaluate": evaluate_cnn_dailymail},
    "GSM8K": {"extract": extract_prompts_gsm8k, "evaluate": evaluate_gsm8k},
    "NLI_MNLI": {"extract": extract_prompts_mnli, "evaluate": evaluate_mnli},
    "SQuAD_v2": {"extract": extract_prompts_squad_v2, "evaluate": evaluate_squad_v2},
    "SST-2": {"extract": extract_prompts_sst2, "evaluate": evaluate_sst2},
    # LongBench - map varied names if strict, or handle dynamically
    "LongBench": {"extract": extract_prompts_longbench, "evaluate": evaluate_longbench}
}

# --- Main Worflow ---

def process_datasets(data_dir, output_csv, limit=None):
    """
    Scans data directory, extracts prompts for known datasets, and saves to CSV.
    """
    all_data = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist.")
        return

    # List directories in data folder
    potential_datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"Found {len(potential_datasets)} potential datasets in {data_dir}")

    for ds_name in potential_datasets:
        handler = None
        
        # Match dataset name to handler
        if ds_name in DATASET_HANDLERS:
            handler = DATASET_HANDLERS[ds_name]
        elif "LongBench" in ds_name:
            handler = DATASET_HANDLERS["LongBench"]
        
        if handler:
            print(f"Processing {ds_name}...")
            try:
                ds_path = os.path.join(data_dir, ds_name)
                dataset = load_from_disk(ds_path)
                
                # Extract
                extracted = handler["extract"](dataset)
                
                # Apply limit if needed (random sampling or first N)
                if limit:
                    extracted = extracted[:limit]
                
                # Update dataset name in case of generic handler usage
                for item in extracted:
                    item["Dataset"] = ds_name
                
                all_data.extend(extracted)
                print(f"  Extracted {len(extracted)} samples.")
                
            except Exception as e:
                print(f"  Error processing {ds_name}: {e}")
        else:
            print(f"  Skipping {ds_name} (No handler defined).")

    # Save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved {len(all_data)} total samples to {output_csv}")
    else:
        print("\nNo data extracted.")

def main():
    parser = argparse.ArgumentParser(description="Dataset Manager: Extract and Evaluate")
    parser.add_argument("--action", type=str, choices=["extract", "evaluate"], default="extract", help="Action to perform")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to datasets directory")
    parser.add_argument("--output_csv", type=str, default="dataset_prompts.csv", help="Output CSV file for extraction")
    parser.add_argument("--limit", type=int, default=100, help="Max samples per dataset to extract")
    # For evaluation inputs
    parser.add_argument("--eval_file", type=str, default="all_experiments_results.csv", help="Input CSV for evaluation")
    
    args = parser.parse_args()
    
    if args.action == "extract":
        process_datasets(args.data_dir, args.output_csv, args.limit)
    
    elif args.action == "evaluate":
        # Example usage of evaluation functions on existing results
        if not os.path.exists(args.eval_file):
            print(f"Evaluation file {args.eval_file} not found.")
            return
            
        print(f"Loading results from {args.eval_file}...")
        df = pd.read_csv(args.eval_file)
        
        results_summary = []
        grouped = df.groupby("Dataset")
        
        for ds_name, group in grouped:
            handler = None
            if ds_name in DATASET_HANDLERS:
                handler = DATASET_HANDLERS[ds_name]
            elif "LongBench" in ds_name:
                handler = DATASET_HANDLERS["LongBench"]
            
            if handler:
                print(f"Evaluating {ds_name}...")
                preds = group["Generated_Output"].fillna("").tolist()
                refs = group["Reference_Output"].fillna("").tolist()
                
                metrics = handler["evaluate"](preds, refs)
                
                for k, v in metrics.items():
                    results_summary.append({
                        "Dataset": ds_name,
                        "Metric": k,
                        "Value": v
                    })
            else:
                print(f"Skipping evaluation for {ds_name} (No handler).")
        
        if results_summary:
            print("\n--- Evaluation Summary ---")
            print(pd.DataFrame(results_summary))

if __name__ == "__main__":
    main()
