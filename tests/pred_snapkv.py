import os
import json
import argparse
import re
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from kvpress import SnapKVPress  #

# --- Configuration ---
# Use the specific Llama 3.1 8B Instruct model
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Standard LongBench Prompt Template
TEMPLATE_0SHOT = """Please read the following text and answer the question below.
<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: "The correct answer is (insert answer here)"."""

def extract_answer(response):
    """Extracts the choice (A, B, C, D) from the model's output."""
    response = response.replace('*', '')
    # Check for explicit format first
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    
    # Fallback checks
    match = re.search(r'The correct answer is ([A-D])', response)
    if match:
        return match.group(1)
        
    return None

def get_pred(data, args, fout):
    # 1. Initialize Tokenizer & Model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Initialize kvpress pipeline
    pipe = pipeline(
        "kv-press-text-generation", 
        model=MODEL_NAME, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True
    )

    # 2. Initialize SnapKV Compression
    # Using SnapKVPress with ratio 0.125 as requested
    press = SnapKVPress(compression_ratio=0.125)

    print(f"Starting inference with context truncation at {args.max_context_length} tokens...")
    
    for item in tqdm(data):
        # --- Context Truncation Logic ---
        context = item['context']
        
        # Tokenize context to check length
        # We assume strict truncation on the context *before* inserting into the template
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        
        if len(context_ids) > args.max_context_length:
            # Truncate to 120k tokens
            context_ids = context_ids[:args.max_context_length]
            context = tokenizer.decode(context_ids)
        
        # --- Prompt Construction ---
        # Fill standard LongBench template
        prompt = TEMPLATE_0SHOT.replace('$DOC$', context.strip())\
                               .replace('$Q$', item['question'].strip())\
                               .replace('$C_A$', item['choice_A'].strip())\
                               .replace('$C_B$', item['choice_B'].strip())\
                               .replace('$C_C$', item['choice_C'].strip())\
                               .replace('$C_D$', item['choice_D'].strip())

        # --- Inference with kvpress ---
        try:
            # We pass the constructed prompt as the 'context' argument to the pipeline.
            # SnapKV naturally preserves the local window (the end of the prompt where the question is).
            output = pipe(
                prompt,
                press=press,
                max_new_tokens=128,
                do_sample=False,  # Greedy decoding for benchmarks
                temperature=1.0,
                return_full_text=False # Try to return only new tokens
            )
            
            # Handle output variations based on pipeline version
            response = ""
            if isinstance(output, dict) and "answer" in output:
                response = output["answer"]
            elif isinstance(output, list) and "generated_text" in output[0]:
                response = output[0]["generated_text"]
            else:
                response = str(output)
                
            # Clean up response if it includes the prompt (common in some pipelines)
            if response.startswith(prompt):
                response = response[len(prompt):]

        except Exception as e:
            print(f"Error on item {item['_id']}: {e}")
            response = ""

        # --- Save Results ---
        item['response'] = response.strip()
        item['pred'] = extract_answer(item['response'])
        item['judge'] = item['pred'] == item['answer']
        # Save truncated context snippet to keep file size manageable
        item['context'] = context[:1000] 
        
        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
        fout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--n_proc", "-n", type=int, default=1, help="Default to 1 for local GPU to avoid OOM")
    parser.add_argument("--max_context_length", type=int, default=120000, help="Truncate context at this token count")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Construct output filename
    out_file = os.path.join(args.save_dir, "Llama-3.1-8B_SnapKV_120k.jsonl")
    print(f"Output to: {out_file}")

    # Load LongBench Dataset
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    # Prepare data structure
    data_all = []
    for item in dataset:
        data_all.append({
            "_id": item["_id"], 
            "domain": item["domain"], 
            "sub_domain": item["sub_domain"], 
            "difficulty": item["difficulty"], 
            "length": item["length"], 
            "question": item["question"], 
            "choice_A": item["choice_A"], 
            "choice_B": item["choice_B"], 
            "choice_C": item["choice_C"], 
            "choice_D": item["choice_D"], 
            "answer": item["answer"], 
            "context": item["context"]
        })

    # Resume capability
    has_data = set()
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            for line in f:
                try:
                    has_data.add(json.loads(line)["_id"])
                except:
                    pass
    
    data_to_process = [x for x in data_all if x["_id"] not in has_data]
    print(f"Total items: {len(data_all)}, Remaining: {len(data_to_process)}")

    fout = open(out_file, 'a', encoding='utf-8')

    # Execution Strategy
    if args.n_proc > 1:
        # Multiprocessing (Only recommended if you have multiple GPUs)
        # Note: 'kvpress' pipeline loading consumes significant VRAM per process
        chunk_size = len(data_to_process) // args.n_proc
        data_subsets = [data_to_process[i::args.n_proc] for i in range(args.n_proc)]
        processes = []
        for rank in range(args.n_proc):
            p = mp.Process(target=get_pred, args=(data_subsets[rank], args, fout))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        # Single process (Recommended for single GPU)
        get_pred(data_to_process, args, fout)

    fout.close()

if __name__ == "__main__":
    main()
