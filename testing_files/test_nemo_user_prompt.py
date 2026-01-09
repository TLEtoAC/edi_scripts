import sys
import os
import time

# Adjust path to find the Nvidia prompt classifier in the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompt_class_dir = os.path.join(parent_dir, "Nvidia prompt class")

if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import temp as nvidia_classifier
except ImportError as e:
    print(f"Error importing nvidia classifier: {e}")
    print(f"Checked path: {prompt_class_dir}")
    print(f"Contents of parent dir: {os.listdir(parent_dir)}")
    sys.exit(1)

# Global cache
NEMO_MODEL = None
NEMO_TOKENIZER = None

def get_nemo_model():
    global NEMO_MODEL, NEMO_TOKENIZER
    if NEMO_MODEL is None or NEMO_TOKENIZER is None:
        print("Loading NeMo Curator model for classification...")
        try:
            NEMO_MODEL, NEMO_TOKENIZER = nvidia_classifier.load_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    return NEMO_MODEL, NEMO_TOKENIZER

def classify_complexity(score):
    if score < 0.35:
        return "Easy"
    elif score < 0.65:
        return "Medium"
    else:
        return "Hard"

def test_user_prompt(prompt):
    print(f"\nAnalyzing Prompt: '{prompt[:50]}...'")
    start_time = time.time()
    
    try:
        model, tokenizer = get_nemo_model()
        result = nvidia_classifier.analyze_prompt(model, tokenizer, prompt)
        
        complexity_score = 0.0
        if "prompt_complexity_score" in result:
             try:
                complexity_score = float(result["prompt_complexity_score"][0])
             except:
                complexity_score = 0.0
        
        category = classify_complexity(complexity_score)
        
        print("\n--- Analysis Results ---")
        for key, value in result.items():
            print(f"{key}: {value}")
            
        print(f"\n--- Classification ---")
        print(f"Complexity Score: {complexity_score:.4f}")
        print(f"Category: {category}")
        print(f"Time Taken: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"Error classifying prompt: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        # Ask for input if not provided
        print("Please enter the prompt you want to test:")
        user_prompt = input("> ")
    
    if user_prompt:
        test_user_prompt(user_prompt)
    else:
        print("No prompt provided.")
