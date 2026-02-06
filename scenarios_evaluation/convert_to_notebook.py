#!/usr/bin/env python3
"""
Script to convert final2.py into a Jupyter notebook with organized cells
"""

import json
import re

def create_code_cell(code):
    """Create a code cell"""
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': code.rstrip().split('\n')
    }

def create_markdown_cell(text):
    """Create a markdown cell"""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': text.rstrip().split('\n')
    }

# Read the Python file
with open('final2.py', 'r') as f:
    content = f.read()

# Initialize notebook structure
notebook = {
    'cells': [],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {'name': 'ipython', 'version': 3},
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.10.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Cell 1: Title
notebook['cells'].append(create_markdown_cell(
"""# EcoPrompt Evaluation - Final2
## Multi-Tier Model Evaluation with Gemini 2.5 Flash & Apple Silicon Support

This notebook implements a comprehensive evaluation framework for comparing different prompt routing and compression strategies across multiple LLM tiers.

**Tiers:**
- **Tier 1:** Phi-3 Mini (Small, Fast)
- **Tier 2:** Mistral 7B (Medium)
- **Tier 3:** Gemini 2.5 Flash (Large, API-based)

**Scenarios:**
- S1: Upper Bound (Tier 3 only)
- S2: Lower Bound (Tier 1 only)
- S3: Compression (Tier 3 + Compression)
- S4: Routing Only
- S5: EcoPrompt (Routing + Compression)"""
))

# Cell 2: Imports
notebook['cells'].append(create_markdown_cell('## 1. Imports and Dependencies'))
notebook['cells'].append(create_code_cell(
"""import argparse
import os
import torch
import pandas as pd
import numpy as np
import time
import csv
import re
import sys
import gc
import signal
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm
from codecarbon import EmissionsTracker
from llmlingua import PromptCompressor
from dotenv import load_dotenv
import google.generativeai as genai"""
))

# Cell 3: Configuration
notebook['cells'].append(create_markdown_cell('## 2. Configuration & Global Setup'))
notebook['cells'].append(create_code_cell(
"""# Load environment variables
load_dotenv()

# Set random seed for reproducibility
torch.manual_seed(42)

# Ensure Nvidia Classifier is in path
current_dir = os.getcwd()
prompt_class_dir = os.path.join(current_dir, "Nvidia prompt class")
if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import nvidia_classifier
except ImportError as e:
    print(f"Warning: Could not import nvidia_classifier from {prompt_class_dir}: {e}")
    nvidia_classifier = None

print("✅ Configuration complete")"""
))

# Extract class definitions
model_manager = re.search(r'(class ModelManager:.*?)(?=\n\nclass )', content, re.DOTALL).group(1)
intelligence_engine = re.search(r'(class IntelligenceEngine:.*?)(?=\n\nclass )', content, re.DOTALL).group(1)
dataset_loader = re.search(r'(class DatasetLoader:.*?)(?=\n\nclass )', content, re.DOTALL).group(1)
evaluator = re.search(r'(class Evaluator:.*?)(?=\n\nclass )', content, re.DOTALL).group(1)
experiment_runner = re.search(r'(class ExperimentRunner:.*?)(?=\n\ndef main)', content, re.DOTALL).group(1)

# Cell 4: ModelManager
notebook['cells'].append(create_markdown_cell(
"""## 3. ModelManager Class
Manages loading, caching, and unloading of LLM models and tokenizers.

**Features:**
- Auto-detects device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
- Loads and caches models for tier1 (Phi-3), tier2 (Mistral), tier3 (Gemini API)
- Handles device-specific optimizations for MPS
- Provides unified text generation interface"""
))
notebook['cells'].append(create_code_cell(model_manager))

# Cell 5: IntelligenceEngine
notebook['cells'].append(create_markdown_cell(
"""## 4. IntelligenceEngine Class
Handles complexity analysis and prompt compression.

**Features:**
- Classifies prompt complexity (Easy/Medium/Hard)
- Routes prompts to appropriate model tier
- Compresses prompts using LLMLingua-2"""
))
notebook['cells'].append(create_code_cell(intelligence_engine))

# Cell 6: DatasetLoader
notebook['cells'].append(create_markdown_cell(
"""## 5. DatasetLoader Class
Handles loading datasets from local files and formatting prompts.

**Supported Datasets:**
- GLUE (MNLI, SST-2)
- SQuAD v2
- CNN/DailyMail
- GSM8K"""
))
notebook['cells'].append(create_code_cell(dataset_loader))

# Cell 7: Evaluator
notebook['cells'].append(create_markdown_cell(
"""## 6. Evaluator Class
Static evaluation metrics for different tasks.

**Metrics:**
- Accuracy (MNLI, SST-2)
- Exact Match (SQuAD, GSM8K)
- ROUGE-L (CNN/DailyMail)"""
))
notebook['cells'].append(create_code_cell(evaluator))

# Cell 8: ExperimentRunner
notebook['cells'].append(create_markdown_cell(
"""## 7. ExperimentRunner Class
Main orchestrator for running experiments.

**Features:**
- Manages multiple scenarios and datasets
- Tracks carbon emissions with CodeCarbon
- Saves results incrementally to prevent data loss
- Handles graceful shutdown on interruption"""
))
notebook['cells'].append(create_code_cell(experiment_runner))

# Cell 9: Main Function
main_func = re.search(r'(def main\(\):.*)', content, re.DOTALL).group(1).rstrip()
notebook['cells'].append(create_markdown_cell('## 8. Main Function\nArgument parsing and experiment initialization.'))
notebook['cells'].append(create_code_cell(main_func))

# Cell 10: Run Configuration
notebook['cells'].append(create_markdown_cell(
"""## 9. Run Experiment
Configure and execute the evaluation.

### Configuration Options:
- `samples`: Number of samples per dataset (default: 5)  
- `datasets`: List of datasets to evaluate (default: all)
- `scenarios`: List of scenarios to run (default: all)
- `no_tracking`: Disable carbon tracking (default: False)
- `output_csv`: Output CSV filename (default: evaluation_scenarios_results.csv)"""
))

notebook['cells'].append(create_code_cell(
"""# Example: Run on GSM8K with 10 samples
class Args:
    samples = 10
    output_csv = "evaluation_scenarios_results.csv"
    no_tracking = False
    datasets = ["gsm8k"]
    scenarios = ["all"]
    data_root = None

args = Args()

# Determine project root
script_dir = os.getcwd()
parent_dir = os.path.dirname(script_dir)

if os.path.exists(os.path.join(parent_dir, "data")):
    project_root = parent_dir
elif os.path.exists(os.path.join(script_dir, "data")):
    project_root = script_dir
else:
    project_root = parent_dir

print(f"[INFO] Using data root: {project_root}")
print(f"[INFO] Looking for data in: {os.path.join(project_root, 'data')}")"""
))

# Cell 11: Execute
notebook['cells'].append(create_markdown_cell('## 10. Execute Experiment\n**Run the cell below to start the evaluation.**'))
notebook['cells'].append(create_code_cell(
"""# Create and run the experiment
runner = ExperimentRunner(args, data_root=project_root)
runner.run()

print("\\n✅ Experiment complete!")
print(f"Results saved to: {args.output_csv}")"""
))

# Cell 12: View Results
notebook['cells'].append(create_markdown_cell('## 11. View Results\nLoad and display the evaluation results.'))
notebook['cells'].append(create_code_cell(
"""# Load results
results_df = pd.read_csv("evaluation_scenarios_results.csv")
summary_df = pd.read_csv("evaluation_scenarios_results_summary.csv")
pivot_df = pd.read_csv("evaluation_scenarios_results_per_prompt.csv")

print("=" * 60)
print("SUMMARY RESULTS")
print("=" * 60)
display(summary_df)

print("\\n" + "=" * 60)
print("DETAILED RESULTS (First 10 rows)")
print("=" * 60)
display(results_df.head(10))

print("\\n" + "=" * 60)
print("PER-PROMPT COMPARISON (First 5 rows)")
print("=" * 60)
display(pivot_df.head(5))"""
))

# Cell 13: Visualization
notebook['cells'].append(create_markdown_cell('## 12. Visualize Results\nCreate plots to compare scenarios.'))
notebook['cells'].append(create_code_cell(
"""import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(14, 6))

# Plot 1: Accuracy by Scenario
plt.subplot(1, 2, 1)
summary_pivot = summary_df.pivot(index='dataset', columns='scenario_name', values='accuracy_score')
summary_pivot.plot(kind='bar', ax=plt.gca())
plt.title('Accuracy by Scenario and Dataset')
plt.xlabel('Dataset')
plt.ylabel('Accuracy Score')
plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Plot 2: Carbon Emissions (if tracked)
plt.subplot(1, 2, 2)
if 'carbon_kg' in results_df.columns and results_df['carbon_kg'].sum() > 0:
    carbon_summary = results_df.groupby('scenario_name')['carbon_kg'].sum()
    carbon_summary.plot(kind='bar', color='green', alpha=0.7, ax=plt.gca())
    plt.title('Carbon Emissions by Scenario')
    plt.xlabel('Scenario')
    plt.ylabel('Total Carbon (kg CO2)')
    plt.xticks(rotation=45, ha='right')
else:
    plt.text(0.5, 0.5, 'Carbon tracking disabled', 
             ha='center', va='center', fontsize=12)
    plt.title('Carbon Emissions')

plt.tight_layout()
plt.show()

print("✅ Visualization complete!")"""
))

# Save notebook
with open('final2.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ Successfully created final2.ipynb")
print(f"Total cells: {len(notebook['cells'])}")
