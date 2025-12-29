import argparse
import os
import time
import torch
import gc
from datasets import load_dataset

from bmrt import RecursiveCompressionEngine

def main(args):
    print("\n" + "="*60)
    print("  Recursive BMRT Inference Multi-Sample Test (v2.0)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Model: {args.model_path}")
    print(f"  - Method: {args.method} (Backend: {args.backend})")
    if args.method == 'lsh':
        print(f"  - LSH Mode: {args.lsh_mode}")
    print(f"  - Budget: {args.budget}")
    print(f"  - Protection Divisor: {args.protection_divisor}")
    print(f"  - Block size: {args.block_size}")
    print(f"  - Top-K: {args.top_k} (Legacy arg, now implicitly global budget)")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print("="*60)

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data ---")
    dataset_name = "RMT-team/babilong-1k-samples"
    try:
        ds = load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # --- 2. Load Engine ---
    print("\n--- 2. Loading RecursiveCompressionEngine ---")
    try:
        engine = RecursiveCompressionEngine(
            model_path=args.model_path,
            selector_type=args.method,
            lsh_mode=args.lsh_mode,
            compression_mode=args.compression_mode,
            backend=args.backend,
            budget=args.budget,
            protection_divisor=args.protection_divisor,
            block_size=args.block_size,
            max_new_tokens=args.max_new_tokens,
            stop_words=args.stop_words.split(',') if args.stop_words else None,
            hybrid_primary=args.hybrid_primary,
            hybrid_secondary=args.hybrid_secondary,
            hybrid_ratio=args.hybrid_ratio,
        )
        print("Engine loaded successfully.")
    except Exception as e:
        print(f"Failed to load engine: {e}")
        return

    # --- 3. Run Inference ---
    print("\n--- 3. Running Inference ---")
    correct = 0
    total = 0
    end_idx = min(args.start_sample_index + args.num_samples, len(ds))

    for i in range(args.start_sample_index, end_idx):
        total += 1
        sample = ds[i]
        context = sample['input']
        query = sample['question']
        target = sample['target']

        print(f"\nProcessing Sample {i}: len(context)={len(context)}")
        # print(f"Query: {query}")
        
        try:
            start = time.time()
            result = engine(prompt_context=context, prompt_query=query)
            duration = time.time() - start
            
            prediction = result['text'][0]
            print(f"Prediction: {prediction}")
            
            if target.lower() in prediction.lower() or prediction.lower() in target.lower():
                correct += 1
                print("Match: PASS")
            else:
                print("Match: FAIL")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    if total > 0:
        print(f"\nAccuracy: {correct/total*100:.2f}% ({correct}/{total})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--method', default='exact', choices=['exact', 'lsh', 'hybrid'])
    parser.add_argument('--lsh_mode', default='frequency_rank', choices=['frequency_rank', 'magicpig_baseline'])
    parser.add_argument('--compression_mode', default='accumulate', choices=['accumulate', 'recursive'])
    parser.add_argument('--backend', default='eager', choices=['eager', 'flash'])
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--protection_divisor', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--stop_words', default='')
    
    # Hybrid args
    parser.add_argument('--hybrid_primary', default='exact')
    parser.add_argument('--hybrid_secondary', default='lsh')
    parser.add_argument('--hybrid_ratio', type=float, default=0.5)
    
    # Dataset args
    parser.add_argument('--dataset_config', default='16k')
    parser.add_argument('--dataset_split', default='qa1')
    parser.add_argument('--start_sample_index', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=0, help="Unused but kept for compatibility")
    
    args = parser.parse_args()
    main(args)
