import time
import csv
from datetime import datetime
from codecarbon import EmissionsTracker
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llmlingua import PromptCompressor
import time
import torch
import sys
import os


# Inlined from nemocurator_test.py
# Add the directory containing the Nvidia prompt classifier code to sys.path
# Using absolute path or relative to be safe.
prompt_class_dir = os.path.abspath("./Nvidia prompt class")
if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import temp as nvidia_classifier
except ImportError as e:
    print(f"Error importing nvidia classifier: {e}")
    # Don't exit, just define a fallback
    nvidia_classifier = None

# Global cache for the specific model used by get_prompt_category
NEMO_MODEL = None
NEMO_TOKENIZER = None

def classify_complexity(score):
    """
    Classifies complexity score into Easy, Medium, Hard.
    Thresholds are illustrative and can be tuned.
    """
    # Assuming score is between 0 and 1, which the weighted sum suggests.
    if score < 0.35:
        return "Easy"
    elif score < 0.65:
        return "Medium"
    else:
        return "Hard"

def get_prompt_category(prompt):
    """
    Analyzes the prompt and returns its complexity category: 'Easy', 'Medium', or 'Hard'.
    Loads the model lazily if not already loaded.
    """
    global NEMO_MODEL, NEMO_TOKENIZER
    
    if nvidia_classifier is None:
        return "Medium"

    if NEMO_MODEL is None or NEMO_TOKENIZER is None:
        print("Loading NeMo Curator model for classification...")
        try:
            NEMO_MODEL, NEMO_TOKENIZER = nvidia_classifier.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to Medium if model fails? Or raise?
            return "Medium" 

    try:
        # analyze_prompt returns a dict with lists inside
        result = nvidia_classifier.analyze_prompt(NEMO_MODEL, NEMO_TOKENIZER, prompt)
        
        complexity_score = 0.0
        if "prompt_complexity_score" in result:
             try:
                complexity_score = float(result["prompt_complexity_score"][0])
             except:
                complexity_score = 0.0
        
        return classify_complexity(complexity_score)
        
    except Exception as e:
        print(f"Error classifying prompt: {e}")
        return "Medium" # Fallback


# Initialize compressor globally or in main to avoid reloading
# For simplicity in this script, we can lazy load or init globally
# But let's init inside main and pass it, or just use a global variable pattern if simple.
# Let's use a global variable for the compressor to keep the function signature simple if that's preferred, 
# or better, refactor to a class or pass 'compressor' argument.
# Given the script structure, let's keep it simple:

COMPRESSOR = None

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

def compress_text(prompt, verbose=True):
    """
    Compresses the given prompt using llm-lingua-2.
    """
    compressor = get_compressor()
    
    try:
        token_count = len(compressor.tokenizer.encode(prompt))
    except:
        token_count = len(prompt) // 4
    
    # Logic provided by user
    slope = 9.5 / 8000
    ratio = 2.5 + (token_count - 2000) * slope
    
    # For testing purposes, if the prompt is short, we might want to force compression 
    # OR we respect the logic.
    # The logic provided: if ratio < 1.0, skip.
    # "Explain transformers..." is very short, so ratio ~ 2.5 + (-1990)*slope ~ 0.13. 
    # It will skip compression.
    
    if ratio < 1.0:
        if verbose:
            print(f"Token Count: {token_count}")
            print(f"Calculated Ratio: {ratio:.2f} (< 1.0). Skipping compression.")
        
        return {
            "compressed_prompt": prompt,
            "rate": 1.0,
            "origin_tokens": token_count,
            "compressed_tokens": token_count,
        }

    if ratio < 2.0:
        ratio = 2.0
    
    rate = 1 / ratio
    
    if verbose:
        print(f"Token Count: {token_count}")
        print(f"Target Ratio: {ratio:.2f}x")
        print(f"Calculated Rate: {rate:.4f}")

    # llmlingua-2 compress_prompt often takes 'rate' or 'target_token'.
    # Checking if force_tokens is supported by llmlingua-2 (usually yes)
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

def load_model1():
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    print(f"Loading model {model_id}...")
    start_time = time.time()

    # Optimize for Apple Silicon (M-series chips)
    # Use 'mps' device if available, otherwise 'cpu'
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16, # Use float16 to reduce memory usage
        device_map=device,
        # trust_remote_code=True, # Disabled to use native transformers implementation
        attn_implementation="eager"
    )

    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    return model, tokenizer

def load_model2():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"Loading model {model_id}...")
    start_time = time.time()

    # Optimize for Apple Silicon (M-series chips)
    # Use 'mps' device if available, otherwise 'cpu'
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16, # Use float16 to reduce memory usage
        device_map=device,
        # trust_remote_code=True, # Usually not needed for TinyLlama native support
        # attn_implementation="eager" # Often helps with compatibility
    )

    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

    return model, tokenizer


def generate_output(model, tokenizer, prompt):
    # Prompt is passed in now
    """
    Generates output for the given prompt using the model.
    """
    
    print(f"Original Prompt Length: {len(prompt)} chars")
    
    # Compress the prompt
    compressed_data = compress_text(prompt)
    compressed_prompt = compressed_data["compressed_prompt"]
    
    print(f"Compressed Prompt Length: {len(compressed_prompt)} chars")
    print(f"Generation using compressed prompt...")
    
    # Use the compressed prompt for generation
    inputs = tokenizer(compressed_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        gen_end = time.time()

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- Response ---")
    print(response)
    print("----------------")
    print(f"Generation took {gen_end - gen_start:.2f} seconds.")
    
    return compressed_data


def log_results(compressed_data, emissions_kg):
    csv_file = "compression_emissions_log.csv"
    file_exists = os.path.isfile(csv_file)
    
    timestamp = datetime.now().isoformat()
    
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Original Tokens", "Compressed Tokens", "Compression Rate", "Emissions (kg CO2eq)"])
            
        writer.writerow([
            timestamp,
            compressed_data["origin_tokens"],
            compressed_data["compressed_tokens"],
            compressed_data["rate"],
            emissions_kg
        ])
    print(f"Logged results to {csv_file}")


def main():
    # Define the prompt here
    prompt = """
    Instruction: summarize the following text.
    
    Context:
    The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived during 1960 during the Eisenhower administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space. Apollo was later dedicated to President John F. Kennedy's national goal of "landing a man on the Moon and returning him safely to the Earth" by the end of the 1960s, which he proposed in an address to Congress on May 25, 1961.
    
    The program was the third US human spaceflight program to fly, preceded by the two-man Project Gemini conceived in 1961 to validate space docking and extra-vehicular activity (EVA) capabilities needed for Apollo, and which it followed in 1968. Apollo ran from 1961 to 1972, with the first crewed flight in 1968. It achieved its goal with the Apollo 11 mission in July 1969, when Neil Armstrong and Buzz Aldrin landed on the Moon while Michael Collins remained in lunar orbit. Five subsequent Apollo missions also landed astronauts on the Moon, the last in December 1972. Throughout these six spaceflights, twelve people walked on the Moon.
    
    Apollo was a major milestone in human history and remains the only time that humans have operated beyond low Earth orbit. The program spurred advances in many areas of technology incidental to rocketry and human spaceflight, including avionics, telecommunications, and computers. Apollo sparked interest in many fields of engineering and is considered by many to be the greatest achievement in the history of mankind.
    
    (Repeating context for length...)
    The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which succeeded in preparing and landing the first humans on the Moon from 1968 to 1972. It was first conceived during 1960 during the Eisenhower administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space. 
    
    The program was the third US human spaceflight program to fly, preceded by the two-man Project Gemini conceived in 1961 to validate space docking and extra-vehicular activity (EVA) capabilities needed for Apollo, and which it followed in 1968. Apollo ran from 1961 to 1972, with the first crewed flight in 1968. It achieved its goal with the Apollo 11 mission in July 1969, when Neil Armstrong and Buzz Aldrin landed on the Moon while Michael Collins remained in lunar orbit.
    
    Apollo was a major milestone in human history and remains the only time that humans have operated beyond low Earth orbit. The program spurred advances in many areas of technology incidental to rocketry and human spaceflight, including avionics, telecommunications, and computers. Apollo sparked interest in many fields of engineering and is considered by many to be the greatest achievement in the history of mankind.
    
    Question: What was the main goal of the Apollo program?
    """

    # Analyze complexity
    print("Analyzing prompt complexity...")
    category = get_prompt_category(prompt)
    print(f"Prompt Category: {category}")
    
    # Select Model
    if category == "Easy":
        print("Selecting Model 1 (tinyllama) for Easy prompt.")
        model, tokenizer = load_model2()
    elif category == "Medium":
        print("Selecting Model 2 (phi3mini) for Medium prompt.")
        model, tokenizer = load_model1()
    else:
        print(f"Selecting Model 2 (TinyLlama) for {category} prompt (Default).")
        model, tokenizer = load_model2()

    print("Initializing CodeCarbon EmissionsTracker...")
    output_dir = "./reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tracker = EmissionsTracker(
        project_name="test_script_demo",
        output_dir=output_dir,
        measure_power_secs=1
    )
    
    print("Starting tracker...")
    tracker.start()
    
    
    compression_stats = None
    compression_stats = None
    try:
        compression_stats = generate_output(model, tokenizer, prompt)
        time.sleep(2) # Sleep to ensure we capture some duration
    finally:
        print("Stopping tracker...")
        emissions = tracker.stop()
        print(f"Emissions: {emissions} kg CO2eq")
        print("Done. Check ./reports/emissions.csv for details.")
        
        if compression_stats:
            log_results(compression_stats, emissions)

if __name__ == "__main__":
    main()






