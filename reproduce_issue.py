from datasets import load_dataset
try:
    print("Loading glue/mnli...")
    load_dataset("glue", "mnli", split="validation_matched[:5]")
    print("Success without revision")
except Exception as e:
    print(f"Failed without revision: {e}")

try:
    print("Loading glue/mnli with revision='main'...")
    load_dataset("glue", "mnli", split="validation_matched[:5]", revision="main")
    print("Success with revision='main'")
except Exception as e:
    print(f"Failed with revision='main': {e}")
