"""
Single-sample test script for Parallel Top-K attention with query-guided token selection.
"""

import argparse
import os
import time
import torch
import gc
from datasets import load_dataset

# FIX: Import the correct class available in the library
from topk_attention import ParallelSmartSummaryProcessor

def main(args):
    print("\n" + "="*60)
    print("  Parallel Top-K Attention Test")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Model: {args.model_path}")
    print(f"  - Block size: {args.block_size}")
    print(f"  - Top-K: {args.top_k}")
    print(f"  - Anchor: {args.anchor_size} | Local: {args.local_window_size}")
    print(f"  - Samples: {args.num_samples}")
    print("="*60)

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data ---")
    
    dataset_name = "RMT-team/babilong-1k-samples"
    print(f"Loading dataset '{dataset_name}', config '{args.dataset_config}'...")
    
    try:
        ds = load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # --- 2. Load the ParallelSmartSummaryProcessor ---
    print("\n--- 2. Loading ParallelSmartSummaryProcessor ---")
    
    # FIX: Instantiate the correct class with correct arguments
    processor = ParallelSmartSummaryProcessor(
        model_path=args.model_path,
        top_k=args.top_k,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
        stop_words=args.stop_words.split(',') if args.stop_words else None,
        anchor_size=args.anchor_size,
        local_window_size=args.local_window_size
    )
    
    print("Processor loaded successfully.")

    # --- 3. Run Inference ---
    correct_predictions = 0
    total_processed_samples = 0
    
    end_index = min(args.start_sample_index + args.num_samples, len(ds))

    for i in range(args.start_sample_index, end_index):
        total_processed_samples += 1
        sample = ds[i]
        
        context = sample['input']
        query = sample['question']
        target = sample['target']

        print(f"\n--- Processing Sample {i} ---")
        print(f"Context length: {len(context)} chars")
        print(f"Query: {query}")
        
        start_time = time.time()
        
        try:
            result = processor(prompt_context=context, prompt_query=query)
            inference_success = True
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            inference_success = False
            result = None
        
        end_time = time.time()
        
        if inference_success and result is not None:
            prediction = result.get('text', [''])[0]
            print(f"\n[Generated Answer]: {prediction}")
            print(f"[Ground Truth]:     {target}")
            
            is_correct = target.lower().strip() in prediction.lower().strip()
            if is_correct:
                correct_predictions += 1
                print("Result: ✓ PASS")
            else:
                print("Result: ✗ FAIL")

        # Cleanup
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Final Report ---
    print("\n" + "="*60)
    if total_processed_samples > 0:
        acc = (correct_predictions / total_processed_samples) * 100
        print(f"Accuracy: {acc:.2f}% ({correct_predictions}/{total_processed_samples})")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--stop_words', type=str, default='')
    
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--top_k', type=int, default=256)
    parser.add_argument('--anchor_size', type=int, default=64, help="Tokens to keep at start of block")
    parser.add_argument('--local_window_size', type=int, default=64, help="Tokens to keep at end of block")
    
    parser.add_argument('--dataset_config', type=str, default='16k')
    parser.add_argument('--dataset_split', type=str, default='qa1')
    parser.add_argument('--start_sample_index', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=1)
    
    args = parser.parse_args()
    main(args)