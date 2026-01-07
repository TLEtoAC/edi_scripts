from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

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

prompt = "Explain transformers in simple terms."

print(f"Generating response for prompt: '{prompt}'")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
