import torch
import csv
import os
import traceback # Added for detailed error logging
from transformers import pipeline
from kvpress import FinchPress
from datasets import load_dataset

# ==========================================
# 1. Configuration & Model Setup
# ==========================================
device = "cuda:0"
model_id = "meta-llama/Llama-3.1-8B-Instruct"
output_csv = "ruler_final_results.csv"

print(f"Loading model: {model_id}...")
pipe = pipeline(
    "kv-press-text-generation", 
    model=model_id, 
    device=device, 
    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16}
)

# --- FIX 1: Lower Chunk Length to 512 ---
# This significantly reduces VRAM usage during the prefill phase.
press = FinchPress(compression_ratio=0.93, chunk_length=512)
press.update_model_and_tokenizer(pipe.model, pipe.tokenizer)

# ==========================================
# 2. Parsing Logic
# ==========================================
def normalize_ruler_output(text):
    # If the model output is an error message, return it as is for the CSV
    if text.startswith("ERROR"):
        return {text}

    text = text.replace("A:", "").strip()
    if "they are:" in text.lower():
        text = text.lower().split("they are:")[-1]
    
    text = text.strip().rstrip(".")
    return {item.strip() for item in text.split(",") if item.strip()}

def get_ground_truth_set(row):
    gt_list = row.get("expected_answer") or row.get("outputs")
    if not gt_list: return set()
    if isinstance(gt_list, list): return set(gt_list)
    clean_str = str(gt_list).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return {item.strip() for item in clean_str.split(",") if item.strip()}

# ==========================================
# 3. Main Loop
# ==========================================
if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "Question", "Prediction", "Ground_Truth", "Correct", "Parsed_Query"])

ds = load_dataset("llamastack/ruler", "vt_128000", split="test")

print("-" * 60)
print("Starting Safe Mode Evaluation (Chunk=512, Truncation=True)...")

for i in range(len(ds)):
    row = ds[i]
    full_text = row["messages"][0]["content"]
    
    # --- Split Logic ---
    split_marker = "Q:"
    if split_marker in full_text:
        parts = full_text.rsplit(split_marker, 1)
        context_part = parts[0].strip()
        question_text = split_marker + parts[1]
    else:
        context_part = full_text[:-500]
        question_text = "Q: " + full_text[-500:]

    # --- FIX 2: Safety Truncation ---
    # 1 token ~= 4 chars. 128k tokens ~= 512,000 chars.
    # We truncate context to 480,000 chars (~120k tokens) to be 100% safe.
    # This leaves space for the Question + Delimiter + Generation.
    if len(context_part) > 480000:
        context_part = context_part[-480000:] # Keep the END of context (usually more relevant)

    # --- Trigger Fix ---
    if not question_text.strip().endswith("A:"):
        question_text = question_text.strip() + "\n\nAnswer: The variables are:"
    
    prompt = context_part + press.delimiter_token + question_text

    # --- Inference ---
    try:
        result = pipe(prompt, question="", press=press, max_new_tokens=100)
        raw_output = result["answer"]
    except Exception:
        # --- FIX 3: Print Full Error ---
        # Capture the full error trace to print to console
        error_msg = traceback.format_exc()
        # Simplify for CSV
        raw_output = f"ERROR: See Console" 
        print(f"\n❌ CRITICAL FAILURE on Sample {i}:")
        print(error_msg)
        print("-" * 20)

    # --- Scoring ---
    pred_set = normalize_ruler_output(raw_output)
    gt_set = get_ground_truth_set(row)
    
    is_correct = (pred_set == gt_set)
    
    # --- Save ---
    with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            i, 
            question_text[:50].replace("\n", " "), 
            str(list(pred_set)), 
            str(list(gt_set)), 
            is_correct,
            question_text 
        ])
    
    status = "✅" if is_correct else "❌"
    print(f"Sample {i}: {status} | Pred: {list(pred_set)}")
