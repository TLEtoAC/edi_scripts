import argparse
import os
import torch
import pandas as pd
import numpy as np
import time
import csv
import re
import sys
import gc
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm
from codecarbon import EmissionsTracker
from llmlingua import PromptCompressor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# Set seed for reproducibility
torch.manual_seed(42)

# Directory Setup for Nvidia Classifier
# Assuming script is run from project root, so 'Nvidia prompt class' is in CWD
current_dir = os.getcwd()
prompt_class_dir = os.path.join(current_dir, "Nvidia prompt class")

if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import nvidia_classifier
except ImportError as e:
    print(f"Warning: Could not import nvidia_classifier from {prompt_class_dir}: {e}")
    nvidia_classifier = None

# --- Global Logic ---

# Models and Tokenizers Cache
MODELS = {
    "tier1": None,  # Phi-3 Mini (Easy)
    "tier2": None,  # Mistral 7B (Medium)
    "tier3": None,  # Llama 2 13B (Hard)
    "nemo": None    # Nvidia Classifier
}
TOKENIZERS = {
    "tier1": None,
    "tier2": None,
    "tier3": None,
    "nemo": None
}
COMPRESSOR = None

def get_device():
    """Returns the primary device (cuda or cpu). Avoids mps as requested."""
    if torch.cuda.is_available():
        return "cuda"
    return "mpu"

# --- 1. Model Loading ---

def load_tier1():
    """Loads Tier 1: Phi-3 Mini"""
    global MODELS, TOKENIZERS
    if MODELS["tier1"] is not None:
        return MODELS["tier1"], TOKENIZERS["tier1"]

    print("Loading Tier 1 (Phi-3 Mini)...")
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    
    try:
        print(f"Checking for local weights for {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True
            )
            print("Loaded Tier 1 (Phi-3) from local cache.")
        except OSError:
            print("Local weights not0 found, downloading/loading from Hub...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True
            )
        # Disable cache to avoid AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'
        # caused by transformers version incompatibility with Phi-3 remote code
        model.generation_config.use_cache = False
        
        MODELS["tier1"] = model
        TOKENIZERS["tier1"] = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Tier 1 (Phi-3): {e}")
        return None, None

def load_tier2():
    """Loads Tier 2: Mistral 7B"""
    global MODELS, TOKENIZERS
    if MODELS["tier2"] is not None:
        return MODELS["tier2"], TOKENIZERS["tier2"]

    print("Loading Tier 2 (Mistral 7B)...")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    try:
        print(f"Checking for local weights for {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto",
                local_files_only=True
            )
            print("Loaded Tier 2 from local cache.")
        except OSError:
            print("Local weights not found, downloading/loading from Hub...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        MODELS["tier2"] = model
        TOKENIZERS["tier2"] = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Tier 2 (Mistral): {e}")
        return None, None

def load_tier3():
    """Loads Tier 3: Llama 2 13B"""
    global MODELS, TOKENIZERS
    if MODELS["tier3"] is not None:
        return MODELS["tier3"], TOKENIZERS["tier3"]

    print("Loading Tier 3 (Llama 2 13B)...")
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Warning: HF_TOKEN not found in environment. Please add it to your .env file.")
    
    try:
        print(f"Checking for local weights for {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto",
                token=token,
                local_files_only=True
            )
            print("Loaded Tier 3 from local cache.")
        except OSError:
            print("Local weights not found, downloading/loading from Hub...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                device_map="auto",
                token=token
            )
        MODELS["tier3"] = model
        TOKENIZERS["tier3"] = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Tier 3 (Llama 2): {e}")
        return None, None

def unload_model(tier):
    """Unloads a model from GPU memory to free space."""
    global MODELS
    if MODELS[tier] is not None:
        print(f"Unloading {tier}...")
        del MODELS[tier]
        del TOKENIZERS[tier]
        torch.cuda.empty_cache()
        gc.collect()
        MODELS[tier] = None
        TOKENIZERS[tier] = None

# --- 2. Classifier Logic ---

def get_nemo_model():
    global MODELS, TOKENIZERS
    if nvidia_classifier is None:
        return None, None
        
    if MODELS["nemo"] is None or TOKENIZERS["nemo"] is None:
        print("Loading NeMo Curator model for classification...")
        try:
            # Note: This might load to GPU. 
            MODELS["nemo"], TOKENIZERS["nemo"] = nvidia_classifier.load_model()
        except Exception as e:
            print(f"Error loading NeMo model: {e}")
            return None, None
    return MODELS["nemo"], TOKENIZERS["nemo"]

def classify_complexity(score):
    if score < 0.35:
        return "Easy"
    elif score < 0.65:
        return "Medium"
    else:
        return "Hard"

def get_prompt_category(prompt):
    """
    Analyzes prompt complexity: 'Easy', 'Medium', or 'Hard'
    """
    if nvidia_classifier is None:
        return "Medium" # Fallback
        
    try:
        model, tokenizer = get_nemo_model()
        if model is None:
            return "Medium"
            
        result = nvidia_classifier.analyze_prompt(model, tokenizer, prompt)
        
        complexity_score = 0.0
        if "prompt_complexity_score" in result:
             try:
                complexity_score = float(result["prompt_complexity_score"][0])
             except:
                complexity_score = 0.0
        
        category = classify_complexity(complexity_score)
        print(f"[DEBUG] Prompt Analysis: Score={complexity_score:.4f}, Category={category}")
        return category
        
    except Exception as e:
        print(f"Error classifying prompt: {e}")
        return "Medium"

def route_prompt(category):
    """
    Routes prompt to appropriate model based on category.
    Easy -> Tier 1 (Phi-3)
    Medium -> Tier 2 (Mistral)
    Hard -> Tier 3 (Llama 2)
    """
    if category == "Easy":
        print(f"[DEBUG] Routing to Tier 1 ({category})")
        return "tier1", load_tier1()
    elif category == "Medium":
        print(f"[DEBUG] Routing to Tier 2 ({category})")
        return "tier2", load_tier2()
    elif category == "Hard":
        print(f"[DEBUG] Routing to Tier 3 ({category})")
        return "tier3", load_tier3()
    else:
        print(f"[DEBUG] Routing Fallback to Tier 2 ({category})")
        return "tier2", load_tier2() # Fallback

# --- 3. Compressor Logic ---

def get_compressor():
    global COMPRESSOR
    if COMPRESSOR is None:
        print("Initializing LLM Lingua-2 Compressor...")
        try:
            # Force CPU for compressor to save GPU VRAM for LLMs if needed
            # Or use 'cuda' if plenty of VRAM. Using 'cpu' is safer for now.
            COMPRESSOR = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu" 
            )
        except Exception as e:
            print(f"Error loading compressor: {e}")
            return None
    return COMPRESSOR

def compress_prompt(prompt):
    """
    Compresses prompt using LLM Lingua.
    """
    compressor = get_compressor()
    if not compressor:
        return {"compressed_prompt": prompt, "rate": 1.0, "initial_tokens": 0, "final_tokens": 0}
        
    try:
        token_count = len(compressor.tokenizer.encode(prompt))
        
        # Heuristic for generic compression logic provided in final.py
        slope = 9.5 / 8000
        ratio = 2.5 + (token_count - 2000) * slope
        
        if ratio < 1.0:
            return {"compressed_prompt": prompt, "rate": 1.0, "initial_tokens": token_count, "final_tokens": token_count}

        if ratio < 2.0:
            ratio = 2.0
            
        rate = 1 / ratio
             
        compressed_result = compressor.compress_prompt(
            prompt, 
            rate=rate, 
            force_tokens=['\n', '?']
        )
        
        if isinstance(compressed_result, dict):
            compressed_text = compressed_result['compressed_prompt']
        else:
            compressed_text = compressed_result
            
        final_tokens = len(compressor.tokenizer.encode(compressed_text))
        
        print(f"[DEBUG] Compression: Rate={rate:.2f}, Init={token_count}, Final={final_tokens}")
        return {
            "compressed_prompt": compressed_text,
            "rate": rate,
            "initial_tokens": token_count,
            "final_tokens": final_tokens
        }

    except Exception as e:
        print(f"Compression error: {e}")
        return {"compressed_prompt": prompt, "rate": 1.0, "initial_tokens": 0, "final_tokens": 0}

# --- 4. Generation ---

def generate(model_weights, prompt):
    print(f"[DEBUG] Generating... Prompt (first 100): {prompt[:100]}...")
    if not model_weights or model_weights[0] is None:
        return "Error: Model not loaded"
        
    model, tokenizer = model_weights
    
    # Use Chat Template for Instruction Models
    messages = [{"role": "user", "content": prompt}]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)
    except Exception:
        # Fallback if chat template fails or not available (though all 3 models should support it)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # Decode only the new tokens
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    print(f"[DEBUG] Generated (first 100): {generated_text.strip()[:100]}...")
    return generated_text.strip()

# --- 5. Evaluation Metrics ---

def evaluate_mnli(pred, label):
    pred = pred.lower()
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    # Check if raw label is int or str
    if isinstance(label, int):
        label_str = mapping.get(label, "")
    else:
        label_str = str(label).lower()
        
    # Simple keyword check
    if "entailment" in pred: p = "entailment"
    elif "neutral" in pred: p = "neutral"
    elif "contradiction" in pred: p = "contradiction"
    else: p = "unknown"
    
    return 1 if p == label_str else 0

def evaluate_sst2(pred, label):
    pred = pred.lower()
    mapping = {0: "negative", 1: "positive"}
    if isinstance(label, int):
        label_str = mapping.get(label, "")
    else:
        label_str = str(label).lower()

    if "positive" in pred: p = "positive"
    elif "negative" in pred: p = "negative"
    else: p = "unknown"
    return 1 if p == label_str else 0

def evaluate_squad(pred, answers):
    # answers is dict: {'text': ['...'], 'answer_start': [...]}
    possible_answers = answers.get('text', [])
    if not possible_answers: return 0
    
    def normalize(s):
        import string
        def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text): return ' '.join(text.split())
        def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
        return white_space_fix(remove_articles(remove_punc(s.lower())))

    pred_norm = normalize(pred)
    return max([1 if normalize(a) == pred_norm else 0 for a in possible_answers])

def evaluate_rouge(pred, ref):
    if not pred: return {"rougeL": 0.0}
    scorer = evaluate.load("rouge")
    results = scorer.compute(predictions=[pred], references=[ref])
    return results

def evaluate_gsm8k(pred, answer_str):
    # answer_str usually ends with #### <number>
    gold = answer_str.split("####")[-1].strip()
    # Extract last number from pred
    pred_nums = re.findall(r'-?\d+\.?\d*', pred)
    pred_num = pred_nums[-1] if pred_nums else ""
    try:
        # Fuzzy float comparison
        return 1 if abs(float(pred_num) - float(gold)) < 1e-5 else 0
    except:
        return 0

# --- Main Scenarios Loop ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="Samples per dataset")
    parser.add_argument("--output_csv", type=str, default="evaluation_scenarios_results.csv")
    parser.add_argument("--no_tracking", action="store_true", help="Disable CodeCarbon (avoids sudo prompts)")
    args = parser.parse_args()
    
    # Datasets
    # Format: (name, subset, split_name)
    target_datasets = [
        ("glue", "mnli", "validation_matched"), 
        ("glue", "sst2", "validation"),
        ("squad_v2", None, "validation"),
        ("cnn_dailymail", "3.0.0", "test"),
        ("gsm8k", "main", "test")
    ]
    
    results = []
    
    tracker = None
    if not args.no_tracking:
        tracker = EmissionsTracker(
            project_name="ecoprompt_scenarios",
            measure_power_secs=1,
            save_to_file=False,
            log_level="error"
        )
    
    # Scenarios Definition
    scenarios = [
        {"id": "S1", "name": "Upper Bound (All -> Tier 3)", "routing": False, "compression": False, "fixed_tier": "tier3"},
        {"id": "S2", "name": "Lower Bound (All -> Tier 1)", "routing": False, "compression": False, "fixed_tier": "tier1"},
        {"id": "S3", "name": "Compression Only (All -> Tier 3 + Comp)", "routing": False, "compression": True, "fixed_tier": "tier3"},
        {"id": "S4", "name": "Routing Only", "routing": True, "compression": False, "fixed_tier": None},
        {"id": "S5", "name": "Proposed System (Router + Comp)", "routing": True, "compression": True, "fixed_tier": None},
    ]

    print(f"Starting Evaluation for {args.samples} samples per dataset...")
    
    for ds_name, subset, split in target_datasets:
        # Local Dataset Loading
        data_files = {}
        if ds_name == "glue" and subset == "mnli":
             data_path = os.path.join(current_dir, "data", "NLI_MNLI", "validation_matched", "*.arrow")
        elif ds_name == "glue" and subset == "sst2":
             data_path = os.path.join(current_dir, "data", "SST-2", "validation", "*.arrow")
        elif ds_name == "squad_v2":
             data_path = os.path.join(current_dir, "data", "SQuAD_v2", "validation", "*.arrow")
        elif ds_name == "cnn_dailymail":
             data_path = os.path.join(current_dir, "data", "CNN_DailyMail", "test", "*.arrow")
        elif ds_name == "gsm8k":
             data_path = os.path.join(current_dir, "data", "GSM8K", "test", "*.arrow")
        else:
             print(f"Unknown local path for {ds_name}/{subset}")
             continue

        print(f"\nProcessing Dataset: {ds_name} (Local Arrow)...")
        try:
            # Load from local arrow files
            # Note: split slicing [0:samples] is done after loading for Arrow datasets usually, 
            # or we load normally and then select. 
            dataset = load_dataset("arrow", data_files=data_path, split="train") 
            # Arrow loading often puts everything in 'train' split by default if not specified differently in data_files dict
            
            # Slice the dataset
            data = dataset.select(range(min(len(dataset), args.samples)))
            
        except Exception as e:
            print(f"Failed to load local dataset {ds_name} from {data_path}: {e}")
            continue

        for i, item in enumerate(tqdm(data, desc=ds_name)):
            print(f"\n[DEBUG] --- Sample {i+1}/{len(data)} ---")
            # Construct Prompt & Reference
            prompt = ""
            reference = ""
            
            if ds_name == "glue" and subset == "mnli":
                prompt = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}\nlabel (entailment, neutral, contradiction):"
                reference = item['label']
            elif ds_name == "glue" and subset == "sst2":
                prompt = f"Sentence: {item['sentence']}\nSentiment (positive, negative):"
                reference = item['label']
            elif ds_name == "squad_v2":
                prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
                reference = item['answers']
            elif ds_name == "cnn_dailymail":
                prompt = f"Summarize:\n{item['article'][:2000]}"
                reference = item['highlights']
            elif ds_name == "gsm8k":
                prompt = f"Question: {item['question']}\nThink step by step:"
                reference = item['answer']

            # Run all 5 scenarios for this specific item
            # To ensure fair comparison, we do this item by item (or we could batch, but item-by-item is easier logic)
            
            # Pre-calculate Router Category once
            category = get_prompt_category(prompt)
            
            for sc in scenarios:
                sid = sc["id"]
                print(f"[DEBUG] Running Scenario: {sid} - {sc['name']}")
                
                # Determine Model
                tier_key = ""
                model_used_name = ""
                model_weights = None
                
                if sc["routing"]:
                    tier_key, model_weights = route_prompt(category)
                    model_used_name = f"{tier_key} ({category})"
                else:
                    tier_key = sc["fixed_tier"]
                    if tier_key == "tier1": model_weights = load_tier1()
                    elif tier_key == "tier2": model_weights = load_tier2()
                    elif tier_key == "tier3": model_weights = load_tier3()
                    model_used_name = tier_key
                
                # Compression
                final_input_prompt = prompt
                comp_stats = {"rate": 1.0, "initial": len(prompt.split()), "final": len(prompt.split())}
                
                if sc["compression"]:
                    # Do compression
                    res = compress_prompt(prompt)
                    final_input_prompt = res["compressed_prompt"]
                    comp_stats["rate"] = res["rate"]
                    comp_stats["initial"] = res["initial_tokens"]
                    comp_stats["final"] = res["final_tokens"]
                
                # Generation & Track Energy
                emissions = 0.0
                output = ""
                
                try:
                    if tracker: tracker.start()
                    output = generate(model_weights, final_input_prompt)
                    if tracker: emissions = tracker.stop()
                except Exception as e:
                    print(f"Error in {sid}: {e}")
                    output = "Error"
                    if tracker and tracker._start_time: # clean up if tracker failed mid-way
                         tracker.stop()
                
                # Evaluation
                score = 0.0
                score_type = "accuracy"
                
                if output != "Error":
                    try:
                        if ds_name == "glue":
                            if subset == "mnli": score = evaluate_mnli(output, reference)
                            elif subset == "sst2": score = evaluate_sst2(output, reference)
                        elif ds_name == "squad_v2":
                            score = evaluate_squad(output, reference)
                            score_type = "EM"
                        elif ds_name == "cnn_dailymail":
                            r = evaluate_rouge(output, reference)
                            score = r.get('rougeL', 0.0)
                            score_type = "ROUGE-L"
                        elif ds_name == "gsm8k":
                            score = evaluate_gsm8k(output, reference)
                            score_type = "EM"
                        
                        print(f"[DEBUG] Evaluation: Score={score} ({score_type})")
                    except:
                        pass # Score remains 0

                # Record Result
                results.append({
                    "scenario_id": sid,
                    "scenario_name": sc["name"],
                    "dataset": ds_name,
                    "sample_index": i,
                    "prompt_complexity": category,
                    "model_used": model_used_name,
                    "original_prompt_len_tokens": comp_stats["initial"],
                    "compressed_prompt_len_tokens": comp_stats["final"],
                    "compression_rate": comp_stats["rate"],
                    "carbon_kg": emissions,
                    "energy_submitted_kwh": 0.0, # Will fill execution energy if available
                    "accuracy_score": score,
                    "score_type": score_type,
                    "output_excerpt": output[:100]
                })
                
                # Capture actual energy consumption from CodeCarbon if available
                if tracker and hasattr(tracker, '_last_emissions_data') and tracker._last_emissions_data:
                     energy_kwh = tracker._last_emissions_data.energy_consumed
                     results[-1]["energy_submitted_kwh"] = energy_kwh

    # Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"\nSuccess! Results saved to {args.output_csv}")
    else:
        print("No results generated.")

    # --- Summary ---
    if results:
        print("\n" + "="*40)
        print("           EVALUATION SUMMARY")
        print("="*40)
        
        # 1. Overall Accuracy per Scenario
        print("\n--- Overall Accuracy by Scenario ---")
        summary_scen = df.groupby(["scenario_id", "scenario_name"])["accuracy_score"].mean().reset_index()
        summary_scen["accuracy_score"] = summary_scen["accuracy_score"].map("{:.2%}".format)
        print(summary_scen.to_string(index=False))
        
        # 2. Detailed Accuracy per Dataset
        print("\n--- Accuracy by Scenario & Dataset ---")
        summary_detail = df.groupby(["scenario_id", "scenario_name", "dataset"])["accuracy_score"].mean().reset_index()
        summary_detail["accuracy_score"] = summary_detail["accuracy_score"].map("{:.2%}".format)
        print(summary_detail.to_string(index=False))
        
        # 3. Average Carbon Emissions (kg) per Scenario
        print("\n--- Avg Carbon Emissions (kg) by Scenario ---")
        summary_carbon = df.groupby(["scenario_id", "scenario_name"])["carbon_kg"].mean().reset_index()
        summary_carbon["carbon_kg"] = summary_carbon["carbon_kg"].map("{:.6f}".format)
        print(summary_carbon.to_string(index=False))

        print("\n" + "="*40)
        
        # Save Summary to CSV
        summary_csv_path = args.output_csv.replace(".csv", "_summary.csv")
        if not summary_csv_path.endswith(".csv"):
             summary_csv_path += "_summary.csv"
             
        summary_detail.to_csv(summary_csv_path, index=False)
        print(f"\nSummary results saved to {summary_csv_path}")

if __name__ == "__main__":
    main()
