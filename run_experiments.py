import time
import csv
import os
import sys
import argparse
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from codecarbon import EmissionsTracker, EmissionsTracker
from llmlingua import PromptCompressor
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Path Setup for Nvidia Classifier ---
# Matches final.py logic
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_class_dir = os.path.abspath("./Nvidia prompt class")
if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import temp as nvidia_classifier
except ImportError as e:
    print(f"Error importing nvidia classifier: {e}")
    nvidia_classifier = None

# --- Globals ---
NEMO_MODEL = None
NEMO_TOKENIZER = None
COMPRESSOR = None

# --- Helpers from final.py ---
def classify_complexity(score):
    if score < 0.35:
        return "Easy"
    elif score < 0.65:
        return "Medium"
    else:
        return "Hard"

def get_prompt_category(prompt):
    global NEMO_MODEL, NEMO_TOKENIZER
    if nvidia_classifier is None:
        raise ImportError("Nvidia classifier not imported/found.")

    if NEMO_MODEL is None or NEMO_TOKENIZER is None:
        print("Loading NeMo Curator model for classification...")
        # Let this raise exception if it fails
        NEMO_MODEL, NEMO_TOKENIZER = nvidia_classifier.load_model()

    # Let this raise exception if it fails
    result = nvidia_classifier.analyze_prompt(NEMO_MODEL, NEMO_TOKENIZER, prompt)
    complexity_score = 0.0
    if "prompt_complexity_score" in result:
             try:
                complexity_score = float(result["prompt_complexity_score"][0])
             except:
                complexity_score = 0.0
    return classify_complexity(complexity_score)

def get_compressor():
    global COMPRESSOR
    if COMPRESSOR is None:
        print("Initializing LLM Lingua-2 Compressor...")
        COMPRESSOR = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
    return COMPRESSOR

def compress_text(prompt, verbose=False):
    compressor = get_compressor()
    # Let this raise exception if tokenization fails
    token_count = len(compressor.tokenizer.encode(prompt))
    
    slope = 9.5 / 8000
    ratio = 2.5 + (token_count - 2000) * slope
    
    if ratio < 1.0:
        return {
            "compressed_prompt": prompt,
            "rate": 1.0,
            "origin_tokens": token_count,
            "compressed_tokens": token_count,
        }

    if ratio < 2.0:
        ratio = 2.0
    
    rate = 1 / ratio
    compressed_result = compressor.compress_prompt(prompt, rate=rate, force_tokens=['\n', '?'])
    
    if isinstance(compressed_result, dict):
        compressed_prompt_text = compressed_result['compressed_prompt']
    else:
        compressed_prompt_text = compressed_result

    return {
        "compressed_prompt": compressed_prompt_text,
        "rate": rate,
        "origin_tokens": token_count,
        "compressed_tokens": len(compressor.tokenizer.encode(compressed_prompt_text)),
    }


# --- Gemini API Helper ---
def call_gemini_api(prompt):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "ERROR: GOOGLE_API_KEY not found in environment variables."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3-flash-preview')
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API Error: {e}"

# --- Model Loading (Replicating final.py quirks) ---
def load_model3():
    # Placeholder for consistency, though we use API directly
    return "gemini", None

def load_model1(): # Phi-3
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    # print(f"Loading {model_id} (Phi-3)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device,
        attn_implementation="eager"
    )
    return model, tokenizer

def load_model2(): # TinyLlama
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # print(f"Loading {model_id} (TinyLlama)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map=device
    )
    return model, tokenizer

# --- Prompt Construction ---
def construct_prompt(dataset_name, example):
    if dataset_name == "CNN_DailyMail":
        return f"Summarize: {example.get('article', '')}"
    elif dataset_name == "GSM8K":
        return f"Question: {example.get('question', '')}\nAnswer:"
    elif dataset_name == "NLI_MNLI":
        return f"Premise: {example.get('premise', '')}\nHypothesis: {example.get('hypothesis', '')}\nLabel:"
    elif dataset_name == "SQuAD_v2":
        return f"Context: {example.get('context', '')}\nQuestion: {example.get('question', '')}\nAnswer:"
    elif dataset_name == "SST-2":
        return f"Sentence: {example.get('sentence', '')}\nSentiment:"
    # LongBench handling might need specfic task logic but for now generic prompt
    elif "LongBench" in dataset_name: 
         return f"Context: {example.get('context', '')}\nInstruction: {example.get('input', '')}\nAnswer:"
    else:
        # Fallback for unknown datasets to avoid crashing, or keep raising error if strict
        # raise ValueError(f"Unknown dataset: {dataset_name}")
        return f"{example}"

def get_reference(dataset_name, example):
    if dataset_name == "CNN_DailyMail":
        return example.get('highlights', '')
    elif dataset_name == "GSM8K":
        return example.get('answer', '')
    elif dataset_name == "NLI_MNLI":
        # MNLI labels: 0: entailment, 1: neutral, 2: contradiction
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
        return label_map.get(example.get('label', -1), str(example.get('label')))
    elif dataset_name == "SQuAD_v2":
        # SQuAD v2 answers is a dict with 'text' list and 'answer_start' list
        # For evaluation we usually take the list of valid texts.
        return example.get('answers', {}).get('text', [])
    elif dataset_name == "SST-2":
        # SST-2 labels: 0: negative, 1: positive
        label_map = {0: "negative", 1: "positive"}
        return label_map.get(example.get('label', -1), str(example.get('label')))
    elif "LongBench" in dataset_name:
        return example.get('answers', []) # LongBench usually has 'answers' list
    else:
        return ""

# --- Main Logic ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=5, help="Max samples per dataset")
    parser.add_argument("--no_tracking", action="store_true", help="Disable CodeCarbon tracking (e.g. to avoid sudo)")
    args = parser.parse_args()

    data_dir = os.path.abspath("./data")
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # List datasets
    # Filter for directories that look like datasets (or just use the list from before)
    # known_datasets = ["CNN_DailyMail", "GSM8K", "NLI_MNLI", "SQuAD_v2", "SST-2"]
    # Dynamic listing:
    datasets_to_run = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith(".")]
    
    # Pre-load models to avoid reloading in loop if possible?
    # Actually final.py loads model purely based on category. 
    # For efficiency, we should keep models loaded if possible, but memory might be tight.
    # We will load them on demand but cache them globally in this script helper?
    # Given the request is to "perform tests in final.py", let's separate the two phases cleanly.
    
    # Cache for models
    models_cache = {} # "phi3" -> (m, t), "tinyllama" -> (m, t)

    def get_model(name):
        if name not in models_cache:
            if name == "phi3":
                models_cache[name] = load_model1()
            elif name == "tinyllama":
                models_cache[name] = load_model2()
            elif name == "gemini":
                models_cache[name] = load_model3()
        return models_cache[name]

    experiments = [
        {"name": "NeMo_Curator", "use_nemo": True, "use_lingua": True, "default_model": "phi3"},
        {"name": "Phi3_Mini_Only", "use_nemo": False, "use_lingua": False, "default_model": "phi3"},
        {"name": "Gemini_Only", "use_nemo": False, "use_lingua": False, "default_model": "gemini"}
    ]
    
    results = []
    
    # Initialize Tracker output dir
    output_dir = "./reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for experiment in experiments:
        exp_name = experiment["name"]
        print(f"--- Starting Experiment: {exp_name} ---")
        
        for ds_name in datasets_to_run:
            ds_path = os.path.join(data_dir, ds_name)
            print(f"Processing Dataset: {ds_name}...")
            
            try:
                # Load dataset
                # Try loading 'test', else 'validation', else 'train'
                dataset = load_from_disk(ds_path)
                if 'test' in dataset:
                    data_split = dataset['test']
                elif 'validation' in dataset:
                    data_split = dataset['validation']
                else:
                    data_split = dataset['train']
                
                # Limit samples
                samples = data_split.select(range(min(len(data_split), args.max_samples)))
                
                for i, example in enumerate(samples):
                    prompt = construct_prompt(ds_name, example)
                    
                    # 1. NeMo Check
                    category = "N/A"
                    if experiment["use_nemo"]:
                        category = get_prompt_category(prompt)
                    
                    # 2. Lingua Compression
                    compressed_prompt = prompt
                    compression_rate = 1.0
                    if experiment["use_lingua"]:
                        comp_res = compress_text(prompt)
                        compressed_prompt = comp_res["compressed_prompt"]
                        compression_rate = comp_res["rate"]
                    
                    # 3. Model Selection
                    selected_model_name = experiment.get("default_model", "phi3") # Default from config
                    if experiment["use_nemo"]:
                        # final.py logic:
                        if category == "Easy":
                            selected_model_name = "tinyllama" # Logic says load_model1 (Phi3) for Easy
                        elif category == "Hard":
                            selected_model_name = "gemini"
                        else:
                            selected_model_name = "phi3" # Logic says load_model2 (TinyLlama) for others
                    
                    model, tokenizer = get_model(selected_model_name)
                    
                    # 4. Generate & Track
                    tracker = None
                    emissions = 0.0
                    
                    if not args.no_tracking:
                        tracker = EmissionsTracker(
                            project_name=f"{exp_name}_{ds_name}",
                            output_dir=output_dir,
                            measure_power_secs=1, # Frequent measurement
                            log_level="error", # Reduce spam
                            save_to_file=False # We handle CSV
                        )
                        tracker.start()
                    
                    start_time = time.time()
                    try:
                        if selected_model_name == "gemini":
                            generated_text = call_gemini_api(compressed_prompt)
                        else:
                            inputs = tokenizer(compressed_prompt, return_tensors="pt").to(model.device)
                            with torch.no_grad():
                                outputs = model.generate(
                                    **inputs, 
                                    max_new_tokens=50, # Keep it short for speed
                                    do_sample=True
                                )
                            # Decode output
                            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Simple post-processing to remove the prompt from the output if it's included
                        # (Phi-3 and others often echo the prompt)
                        if generated_text.startswith(compressed_prompt):
                            generated_text = generated_text[len(compressed_prompt):].strip()
                        elif generated_text.startswith(prompt): # Logic check
                            generated_text = generated_text[len(prompt):].strip()
                        
                        # Further cleanup if needed (e.g., stopping at newlines for some tasks)
                        if ds_name in ["SST-2", "NLI_MNLI"]:
                            generated_text = generated_text.split('\n')[0].strip().lower()
                            
                    except Exception as e:
                        print(f"Inference Error: {e}")
                        generated_text = f"ERROR: {e}"
                    
                    if tracker:
                        emissions = tracker.stop()
                    end_time = time.time()
                    
                    # 5. Log
                    row = {
                        "Experiment": exp_name,
                        "Dataset": ds_name,
                        "Sample_ID": i,
                        "Original_Length": len(prompt),
                        "Compressed_Length": len(compressed_prompt),
                        "Compression_Rate": compression_rate,
                        "Prompt_Category": category,
                        "Model_Used": selected_model_name,
                        "Time_Sec": end_time - start_time,
                        "Emissions_kg": emissions,
                        "Original_Prompt": prompt,
                        "Compressed_Prompt": compressed_prompt,
                        "Generated_Output": generated_text,
                        "Reference_Output": get_reference(ds_name, example)
                    }
                    results.append(row)

                    # print(f"  Sample {i}: {category}, {selected_model_name}, {emissions:.2e} kgCO2e")

            except Exception as e:
                print(f"Skipping dataset {ds_name} due to error: {e}")
    
    # Save to CSV
    if results:
        csv_file = "all_experiments_results.csv"
        keys = results[0].keys()
        with open(csv_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"Results saved to {csv_file}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
