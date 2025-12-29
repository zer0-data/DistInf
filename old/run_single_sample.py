import argparse
import torch
import time
from datasets import load_dataset

# Import the custom model class from your modified model.py file
from old.model import CustomAccuracyModel

def main(args):
    """
    Main function to test SqueezedAttention-style inference on a single sample.
    """
    print("--- 1. Loading Data ---")
    
    dataset_name = "RMT-team/babilong-1k-samples"
    
    print(f"Loading dataset '{dataset_name}', config '{args.dataset_config}', split '{args.dataset_split}'...")
    try:
        ds = load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    sample = ds[args.sample_index]
    
    context = sample['input']
    query = sample['question']
    target = sample['target']

    print("\n--- Sample Details ---")
    print(f"Context length: {len(context)} characters")
    print(f"Query: {query}")
    print(f"Ground Truth: {target}")
    print("-" * 40)

    # 2. Load Model
    print("\n--- 2. Loading Model ---")
    model = CustomAccuracyModel(
        path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        k_summary_size=args.k_summary_size,
        summary_method=args.summary_method,
        pruning_percent=args.pruning_percent,
        use_cosine_similarity=args.use_cosine_similarity,
        multi_layer_aggregation=args.multi_layer_aggregation,
        num_layers_for_clustering=args.num_layers_for_clustering,
    )

    # 3. Run Inference
    print("\n--- 3. Running Inference ---")
    start_time = time.time()
    
    result = model(prompt_context=context, prompt_query=query)
    
    end_time = time.time()
    prediction = result['text'][0]

    # 4. Display Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Time: {end_time - start_time:.2f} seconds")
    print(f"\nQuery: {query}")
    print(f"\nGround Truth: {target}")
    print(f"\nPrediction: {prediction}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test SqueezedAttention-style inference")
    
    parser.add_argument('--model_path', required=True, help='Model path')
    parser.add_argument('--block_size', type=int, default=4096, help='Block size')
    parser.add_argument('--k_summary_size', type=int, default=128, help='Summary size (for top_k method)')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Max tokens to generate')
    
    # Summary method
    parser.add_argument('--summary_method', type=str, default='kmeans', choices=['top_k', 'kmeans'])
    parser.add_argument('--pruning_percent', type=float, default=90.0, 
                        help='Percent of tokens to prune (0-100). E.g., 90 = keep 10%%')
    
    # K-means parameters
    parser.add_argument('--use_cosine_similarity', action='store_true', default=True)
    parser.add_argument('--no_cosine_similarity', action='store_false', dest='use_cosine_similarity')
    parser.add_argument('--multi_layer_aggregation', action='store_true', default=True)
    parser.add_argument('--no_multi_layer', action='store_false', dest='multi_layer_aggregation')
    parser.add_argument('--num_layers_for_clustering', type=int, default=4, 
                        help='Number of layers to aggregate for clustering')
    
    # Dataset
    parser.add_argument('--dataset_config', type=str, default='16k')
    parser.add_argument('--dataset_split', type=str, default='qa1')
    parser.add_argument('--sample_index', type=int, default=0)
    
    args = parser.parse_args()
    main(args)