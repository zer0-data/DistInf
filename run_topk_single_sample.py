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
import torch # added for cuda memory management
import gc # added for garbage collection
from datasets import load_dataset

# Import the SequentialTopKProcessor from topk_attention.py
from topk_attention import SequentialTopKProcessor


def main(args):
    """
    Main function to load data, load the model, run inference on a single sample,
    and print the results for comparison.
    """
    print("\n" + "="*60)
    print("  Sequential Top-K Attention Multiple Samples Test")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Model: {args.model_path}")
    print(f"  - Block size: {args.block_size}")
    print(f"  - Top-K: {args.top_k}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Number of samples to process: {args.num_samples}")
    print(f"  - Starting sample index: {args.start_sample_index}")
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

    # --- 2. Load the SequentialTopKProcessor (once for efficiency) ---
    print("\n--- 2. Loading SequentialTopKProcessor ---")
    
    processor = SequentialTopKProcessor(
        model_path=args.model_path,
        top_k=args.top_k,
        block_size=args.block_size,
        max_new_tokens=args.max_new_tokens,
        stop_words=args.stop_words.split(',') if args.stop_words else None,
    )
    
    print("Processor loaded successfully.")

    # --- 3. Run Inference on Multiple Samples ---
    print("\n--- 3. Running Inference Across Samples ---")
    
    correct_predictions = 0
    total_processed_samples = 0
    
    end_index = min(args.start_sample_index + args.num_samples, len(ds))

    for i in range(args.start_sample_index, end_index):
        total_processed_samples += 1
        
        sample_idx = i
        if sample_idx >= len(ds):
            print(f"Warning: Sample index {sample_idx} out of range. Stopping.")
            break
            
        sample = ds[sample_idx]
        
        context = sample['input']
        query = sample['question']
        target = sample['target']

        print(f"\n--- Processing Sample {sample_idx} ---")
        print(f"Context length: {len(context)} characters")
        print(f"Query: {query}")
        print(f"Ground Truth Target: {target}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            result = processor(prompt_context=context, prompt_query=query)
            inference_success = True
        except Exception as e:
            print(f"\nError during inference for sample {sample_idx}: {e}")
            import traceback
            traceback.print_exc()
            inference_success = False
            result = None
        
        end_time = time.time()
        
        # --- 4. Display Results and Calculate Accuracy ---
        if inference_success and result is not None:
            prediction = result.get('text', [''])[0] if isinstance(result, dict) else str(result)
            
            print(f"\nInference completed in {end_time - start_time:.2f} seconds for sample {sample_idx}.")

            print(f"\n[Model's Generated Answer]:\n{prediction}")
            
            # Simple accuracy check
            target_lower = target.lower().strip()
            prediction_lower = prediction.lower().strip()
            is_correct = target_lower in prediction_lower or prediction_lower in target_lower
            
            if is_correct:
                correct_predictions += 1
                print(f"\n[Match Check]: ✓ PASS")
            else:
                print(f"\n[Match Check]: ✗ FAIL")
        else:
            print(f"\nSkipping accuracy check for sample {sample_idx} due to inference error.")

        print("\n" + "="*60)
        
        # --- 5. Clear Memory After Each Sample ---
        del result
        del prediction
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared.")
        print("-" * 60) # Separator for samples

    # --- Final Accuracy Report ---
    print("\n" + "="*60)
    print("      OVERALL ACCURACY REPORT")
    print("="*60)
    if total_processed_samples > 0:
        accuracy = (correct_predictions / total_processed_samples) * 100
        print(f"Total samples processed: {total_processed_samples}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No samples were processed.")
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
        default=128, 
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
        '--start_sample_index', # Renamed from sample_index
        type=int,
        default=0,
        help='Starting index of the sample to test (default: 0)'
    )
    parser.add_argument(
        '--num_samples', # New argument
        type=int,
        default=1000,
        help='Number of samples to process starting from --start_sample_index (default: 1)'
    )
    
    args = parser.parse_args()
    
<<<<<<< HEAD
    main(args)
=======
    main(args)
>>>>>>> 5533a9f94d8e638714bc6cb8d38ec9bb0f31da61
