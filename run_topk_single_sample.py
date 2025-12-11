"""
Single-sample test script for Sequential Top-K attention with query-guided token selection.

This script tests the pipeline:
  Phase 1: Block + Query → Select Top-K from Block (per block)
  Phase 2: Build KV cache sequentially with summaries
  Phase 3: Generate response

Usage:
    python run_topk_single_sample.py --model_path <path_to_model>

Example:
    python run_topk_single_sample.py \
        --model_path meta-llama/Llama-2-7b-hf \
        --top_k 256 \
        --block_size 2048 \
        --max_new_tokens 100
"""

import argparse
import os
import time
from datasets import load_dataset

# Import the SequentialTopKProcessor from topk_attention.py
from topk_attention import SequentialTopKProcessor


def main(args):
    """
    Main function to load data, load the model, run inference on a single sample,
    and print the results for comparison.
    """
    print("\n" + "="*60)
    print("  Sequential Top-K Attention Single Sample Test")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Model: {args.model_path}")
    print(f"  - Block size: {args.block_size}")
    print(f"  - Top-K: {args.top_k}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print("="*60)

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data ---")
    
    dataset_name = "RMT-team/babilong-1k-samples"
    config_name = args.dataset_config
    split_name = args.dataset_split
    
    print(f"Loading dataset '{dataset_name}', config '{config_name}', split '{split_name}'...")
    
    try:
        ds = load_dataset(dataset_name, config_name, split=split_name)
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return

    # Select the sample from the dataset
    sample_idx = args.sample_index
    if sample_idx >= len(ds):
        print(f"Sample index {sample_idx} out of range. Dataset has {len(ds)} samples.")
        return
        
    sample = ds[sample_idx]
    
    context = sample['input']
    query = sample['question']
    target = sample['target']

    print("\n--- Sample Details ---")
    print(f"Sample index: {sample_idx}")
    print(f"Context length: {len(context)} characters")
    print(f"Context (first 300 chars): {context[:300]}...")
    print(f"Query: {query}")
    print(f"Ground Truth Target: {target}")
    print("-" * 40)

    # --- 2. Load the SequentialTopKProcessor ---
    print("\n--- 2. Loading SequentialTopKProcessor ---")
    
    processor = SequentialTopKProcessor(
        model_path=args.model_path,
        top_k=args.top_k,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
        stop_words=args.stop_words.split(',') if args.stop_words else None,
    )
    
    print("Processor loaded successfully.")

    # --- 3. Run Inference on the Single Sample ---
    print("\n--- 3. Running Inference ---")
    
    start_time = time.time()
    
    try:
        result = processor(prompt_context=context, prompt_query=query)
        inference_success = True
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
        inference_success = False
        result = None
    
    end_time = time.time()
    
    # --- 4. Display Results ---
    if inference_success and result is not None:
        prediction = result.get('text', [''])[0] if isinstance(result, dict) else str(result)
        
        print(f"\nInference completed in {end_time - start_time:.2f} seconds.")

        print("\n" + "="*60)
        print("      FINAL RESULTS COMPARISON")
        print("="*60)
        print(f"\n[Query]: {query}")
        print("\n" + "-"*40)
        print(f"[Ground Truth Answer]:\n{target}")
        print("\n" + "-"*40)
        print(f"[Model's Generated Answer]:\n{prediction}")
        print("\n" + "="*60)
        
        # Simple accuracy check
        target_lower = target.lower().strip()
        prediction_lower = prediction.lower().strip()
        is_correct = target_lower in prediction_lower or prediction_lower in target_lower
        print(f"\n[Match Check]: {'✓ PASS' if is_correct else '✗ FAIL'}")
        print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run single-sample inference with Sequential Top-K Processor on the babilong dataset."
    )
    
    # Model arguments
    parser.add_argument(
        '--model_path', 
        required=True, 
        help='Path to the model checkpoint (e.g., "meta-llama/Llama-2-7b-hf")'
    )
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=100, 
        help='Maximum number of new tokens to generate for the answer'
    )
    parser.add_argument(
        '--stop_words',
        type=str,
        default='',
        help='Comma-separated stop words for generation'
    )
    
    # Top-K arguments
    parser.add_argument(
        '--block_size', 
        type=int, 
        default=2048, 
        help='Block size for context processing'
    )
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=256, 
        help='Number of tokens to select per block via attention accumulation'
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset_config',
        type=str,
        default='16k',
        help='Dataset configuration (e.g., "16k", "32k", "64k", "128k")'
    )
    parser.add_argument(
        '--dataset_split',
        type=str,
        default='qa1',
        help='Dataset split (e.g., "qa1", "qa2", etc.)'
    )
    parser.add_argument(
        '--sample_index',
        type=int,
        default=0,
        help='Index of the sample to test (default: 0)'
    )
    
    args = parser.parse_args()
    
    main(args)
