"""
"""Single-sample test script for StarAttentionModel with Top-K attention selection.

This script tests block-wise attention accumulation across all layers,
selecting the top-K most attended tokens per block for sparse KV cache.

Usage (requires distributed setup even for single GPU):
    torchrun --nproc_per_node=1 run_topk_single_sample.py --model_path <path_to_model>

For multi-GPU:
    torchrun --nproc_per_node=<num_gpus> run_topk_single_sample.py --model_path <path_to_model>
"""

import argparse
import os
import time
import torch
import torch.distributed as dist
from datasets import load_dataset

# Import the StarAttentionModel from model.py
from model import StarAttentionModel


def init_distributed():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f'[init_distributed] Rank: {rank}, World size: {world_size}')
    else:
        # Fallback for non-distributed run (will likely fail for StarAttentionModel)
        print("Warning: Running without distributed environment.")
        print("StarAttentionModel requires torchrun. Use:")
        print("  torchrun --nproc_per_node=1 run_topk_single_sample.py --model_path <path>")
        rank = 0
        world_size = 1
    return rank, world_size


def main(args):
    """
    Main function to load data, load the model, run inference on a single sample,
    and print the results for comparison.
    """
    rank, world_size = init_distributed()
    
    if rank == 0:
        print("\n" + "="*60)
        print("  Top-K Attention Single Sample Test")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  - Model: {args.model_path}")
        print(f"  - Block size: {args.block_size}")
        print(f"  - Num blocks: {args.num_blocks}")
        print(f"  - Top-K: {args.top_k}")
        print(f"  - Max new tokens: {args.max_new_tokens}")
        print(f"  - World size: {world_size}")
        print("="*60)

    # --- 1. Load Data (only rank 0 prints) ---
    if rank == 0:
        print("\n--- 1. Loading Data ---")
    
    # Load the specific dataset, configuration, and split
    dataset_name = "RMT-team/babilong-1k-samples"
    config_name = args.dataset_config
    split_name = args.dataset_split
    
    if rank == 0:
        print(f"Loading dataset '{dataset_name}', config '{config_name}', split '{split_name}'...")
    
    try:
        ds = load_dataset(dataset_name, config_name, split=split_name)
    except Exception as e:
        if rank == 0:
            print(f"Failed to load dataset. Error: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # Select the sample from the dataset
    sample_idx = args.sample_index
    if sample_idx >= len(ds):
        if rank == 0:
            print(f"Sample index {sample_idx} out of range. Dataset has {len(ds)} samples.")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
        
    sample = ds[sample_idx]
    
    context = sample['input']
    query = sample['question']
    target = sample['target']

    if rank == 0:
        print("\n--- Sample Details ---")
        print(f"Sample index: {sample_idx}")
        print(f"Context length: {len(context)} characters")
        print(f"Context (first 300 chars): {context[:300]}...")
        print(f"Query: {query}")
        print(f"Ground Truth Target: {target}")
        print("-" * 40)

    # --- 2. Load the StarAttentionModel with Top-K attention ---
    if rank == 0:
        print("\n--- 2. Loading StarAttentionModel with Top-K Attention ---")
    
    # Synchronize before model loading
    if dist.is_initialized():
        dist.barrier()
    
    model = StarAttentionModel(
        path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        stop_words=args.stop_words.split(',') if args.stop_words else None,
        top_k=args.top_k,
        selection_layers=[],  # Not used in new accumulation approach
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        anchor_block_size=args.anchor_block_size,
    )
    
    if rank == 0:
        print("Model loaded successfully.")
        print(f"  - Top-K attention enabled with top_k={args.top_k}")
        print(f"  - Block-wise accumulation across ALL layers")

    # --- 3. Run Inference on the Single Sample ---
    if rank == 0:
        print("\n--- 3. Running Inference ---")
        print("The model will now run sequential Top-K prefill followed by generation...")
    
    # Synchronize before inference
    if dist.is_initialized():
        dist.barrier()
    
    start_time = time.time()
    
    try:
        result = model(prompt_context=context, prompt_query=query)
        inference_success = True
    except Exception as e:
        if rank == 0:
            print(f"\nError during inference: {e}")
            import traceback
            traceback.print_exc()
        inference_success = False
        result = None
    
    end_time = time.time()
    
    # --- 4. Display Results (only rank 0) ---
    if rank == 0 and inference_success and result is not None:
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

    # Cleanup
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run single-sample inference with StarAttentionModel + Top-K prefill on the babilong dataset."
    )
    
    # Model arguments
    parser.add_argument(
        '--model_path', 
        required=True, 
        help='Path to the model checkpoint (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")'
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
    
    # Top-K prefill arguments
    parser.add_argument(
        '--block_size', 
        type=int, 
        default=-1, 
        help='Block size for context processing (-1 to use num_blocks instead)'
    )
    parser.add_argument(
        '--num_blocks',
        type=int,
        default=4,
        help='Number of blocks to split context into (used if block_size=-1)'
    )
    parser.add_argument(
        '--anchor_block_size',
        type=int,
        default=-1,
        help='Anchor block size (-1 for automatic)'
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
    
    # Validate model path
    if not os.path.exists(args.model_path) and not args.model_path.startswith(('meta-llama/', 'huggingface/')):
        print(f"Warning: Model path '{args.model_path}' does not exist locally.")
        print("Will attempt to load from HuggingFace Hub...")
    
    main(args)
