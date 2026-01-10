import os, csv, json
import argparse
import time
from tqdm import tqdm
from datasets import load_dataset
import re
import torch
import gc
import sys


from bmrt import RecursiveCompressionEngine

# Remove OpenAI/network related imports and globals
# model_map and maxlen_map is replaced by direct args or simple logic if needed, 
# but we will keep the structure of loading dataset and extracting answer.

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def main():
    # --- Argument Parsing (Merged from pred.py and run_single_sample.py) ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model_path", "-m", type=str, default="gradientai/Llama-3-8B-Instruct-Gradient-1048k") # Default updated
    parser.add_argument("--start_sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=999999) # Default to all key it simple
    
    # BMRT specific args
    parser.add_argument('--method', default='hybrid', choices=['exact', 'lsh', 'hybrid'])
    parser.add_argument('--lsh_mode', default='frequency_rank', choices=['frequency_rank', 'magicpig_baseline'])
    parser.add_argument('--mode', default='partitioned_centroid', choices=['l2', 'max_sim', 'mahalanobis', 'partitioned_centroid', 'none'], help='Tie-breaker mode for LSHSelector')
    parser.add_argument('--compression_mode', default='accumulate', choices=['accumulate', 'recursive'])
    parser.add_argument('--backend', default='eager', choices=['eager', 'flash'])
    parser.add_argument('--compression_ratio', type=float, default=0.125, help='Ratio of tokens to retain (retain/total)')
    parser.add_argument('--protection_divisor', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=64) # Adjusted valid default
    parser.add_argument('--stop_words', default='')
    
    # LSH args
    parser.add_argument('--num_bits', type=int, default=10, help='Number of bits per LSH hash')
    parser.add_argument('--num_tables', type=int, default=8, help='Number of LSH hash tables')
    
    # Hybrid args
    parser.add_argument('--hybrid_primary', default='exact')
    parser.add_argument('--hybrid_secondary', default='lsh')
    parser.add_argument('--hybrid_ratio', type=float, default=0.75)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    # Output file definition
    model_name = args.model_path.split("/")[-1]
    out_file = os.path.join(args.save_dir, f"{model_name}_{args.method}_{args.compression_ratio}.jsonl")
    
    # Load Dataset
    print(f"Loading LongBench-v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train')
    
    # Filter/Select data
    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]
    
    # Check existing data to resume
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            for line in f:
                try:
                    has_data[json.loads(line)["_id"]] = 0
                except:
                    pass
    
    fout = open(out_file, 'a', encoding='utf-8')
    
    # Filter data to process
    data_to_process = []
    for item in data_all:
        if item["_id"] not in has_data:
            data_to_process.append(item)
            
    # Apply limit
    data_to_process = data_to_process[args.start_sample_index : args.start_sample_index + args.num_samples]
    
    print(f"Total samples to process: {len(data_to_process)}")

    # Helper function to build engine
    def build_engine(budget):
        try:
            eng = RecursiveCompressionEngine(
                model_path=args.model_path,
                selector_type=args.method,
                lsh_mode=args.lsh_mode,
                selector_mode=args.mode,
                compression_mode=args.compression_mode,
                backend=args.backend,
                budget=budget,
                protection_divisor=args.protection_divisor,
                block_size=args.block_size,
                max_new_tokens=args.max_new_tokens,
                stop_words=args.stop_words.split(',') if args.stop_words else None,
                num_bits=args.num_bits,
                num_tables=args.num_tables,
                hybrid_primary=args.hybrid_primary,
                hybrid_secondary=args.hybrid_secondary,
                hybrid_ratio=args.hybrid_ratio,
            )
            return eng
        except Exception as e:
            print(f"Failed to load engine: {e}")
            return None

    # Processing Loop
    for i, item in enumerate(tqdm(data_to_process, desc="Processing")):
        context = item['context']
        question = item['question']
        
        # Budget Calculation
        try:
            context_len = int(item['length']) # Provided by dataset, ensuring it's an integer
        except ValueError:
            print(f"Warning: item['length'] is not a valid integer: {item['length']}. Defaulting to 8192 tokens for context_len.")
            context_len = 131072 # Default to a large value for 'long' contexts
            
        # Logic from run_single_sample.py
        if args.compression_mode == 'recursive':
             budget = int(args.compression_ratio * context_len)
        else: # accumulate
             budget = int((2 * args.compression_ratio * args.block_size * context_len) / (args.block_size + context_len))
        
        # Ensure budget is at least something minimal
        budget = max(budget, 512) 

        # print(f"Sample {item['_id']}: Length={context_len}, Budget={budget}")

        engine = build_engine(budget)
        if engine is None:
            print(f"Skipping {item['_id']} due to engine failure.")
            continue

        try:
            # Prepare prompt
            # Original pred.py logic for prompt construction:
            # prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip())...
            # We should follow how run_single_sample does it: prompt_context, prompt_query
            # Wait, pred.py constructs a specific prompt with choices. 
            # run_single_sample takes query and context separate.
            # We need to format the query part to include choices as pred.py does.
            
            # Reconstruct the question part with choices
            formatted_question = f"{item['question']}\n\nA. {item['choice_A']}\nB. {item['choice_B']}\nC. {item['choice_C']}\nD. {item['choice_D']}\n\nAnswer:"
            
            # Run inference
            result = engine(prompt_context=context, prompt_query=formatted_question)
            
            response = result['text'][0]
            
            # Process output
            item['response'] = response
            item['pred'] = extract_answer(response)
            item['judge'] = item['pred'] == item['answer']
            item['context'] = context[:1000] # Truncate for save
            
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()

        except Exception as e:
            print(f"Error processing {item['_id']}: {e}")
            import traceback
            traceback.print_exc()
        
        # Cleanup
        try:
            engine.cleanup()
        except:
            pass
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    fout.close()

if __name__ == "__main__":
    main()
