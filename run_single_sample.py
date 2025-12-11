import argparse
import torch
import time
from datasets import load_dataset

# Import the custom model class from your modified model.py file
from model import CustomAccuracyModel

def main(args):
    """
    Main function to load data, load the model, run inference on a single sample,
    and print the results for comparison.
    """
    print("--- 1. Loading Data ---")
    
    # Load the specific dataset, configuration, and split
    dataset_name = "RMT-team/babilong-1k-samples"
    config_name = "16k"
    split_name = "qa1"
    
    print(f"Loading dataset '{dataset_name}', config '{config_name}', split '{split_name}'...")
    try:
        ds = load_dataset(dataset_name, config_name, split=split_name)
    except Exception as e:
        print(f"Failed to load dataset. Please ensure you have an internet connection and the 'datasets' library is installed.")
        print(f"Error: {e}")
        return

    # Select the first sample from the dataset for testing
    sample = ds[0]
    
    context = sample['input']
    query = sample['question']
    target = sample['target']

    print("\n--- Sample Details ---")
    print(f"Context (first 200 chars): {context[:200]}...")
    print(f"Query: {query}")
    print(f"Ground Truth Target: {target}")
    print("-" * 22)

    # 2. Load the Custom Accuracy Model
    print("\n--- 2. Loading Model ---")
    print(f"Using summary method: {args.summary_method}")
    model = CustomAccuracyModel(
        path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        k_summary_size=args.k_summary_size,
        summary_method=args.summary_method,
        pruning_percent=args.pruning_percent,
        # New k-means improvement parameters
        use_cosine_similarity=args.use_cosine_similarity,
        multi_layer_aggregation=args.multi_layer_aggregation,
        query_guided=args.query_guided,
        query_weight=args.query_weight,
    )
    print("Model loaded successfully.")

    # 3. Run Inference on the Single Sample
    print("\n--- 3. Running Inference ---")
    print("The model will now print detailed debug logs for each step...")
    start_time = time.time()
    
    result = model(prompt_context=context, prompt_query=query)
    
    end_time = time.time()
    prediction = result['text'][0]
    print(f"\nInference completed in {end_time - start_time:.2f} seconds.")

    # 4. Display Final Comparison
    print("\n\n" + "="*30)
    print("      FINAL RESULTS COMPARISON")
    print("="*30)
    print(f"\n[Query]: {query}")
    print("\n" + "-"*30)
    print(f"[Ground Truth Answer]:\n{target}")
    print("\n" + "-"*30)
    print(f"[Model's Generated Answer]:\n{prediction}")
    print("\n" + "="*30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run single-sample inference with the CustomAccuracyModel on the babilong dataset."
    )
    parser.add_argument(
        '--model_path', 
        required=True, 
        help='Path to the model checkpoint (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")'
    )
    parser.add_argument(
        '--block_size', 
        type=int, 
        default=4096, 
        help='Block size for context processing'
    )
    parser.add_argument(
        '--k_summary_size', 
        type=int, 
        default=128, 
        help='Number of summary tokens to extract per block'
    )
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=100, 
        help='Maximum number of new tokens to generate for the answer'
    )
    parser.add_argument(
        '--summary_method',
        type=str,
        default='top_k',
        choices=['top_k', 'kmeans'],
        help='Method for summarizing context blocks: "top_k" (attention-based) or "kmeans" (clustering-based)'
    )
    parser.add_argument(
        '--pruning_percent',
        type=float,
        default=90.0,
        help='Percentage of tokens to prune (0-100). Only used with kmeans method. E.g., 90 means keep 10%% of tokens.'
    )
    
    # New arguments for k-means improvements
    parser.add_argument(
        '--use_cosine_similarity',
        action='store_true',
        default=True,
        help='Use cosine similarity instead of Euclidean distance for k-means (default: True)'
    )
    parser.add_argument(
        '--no_cosine_similarity',
        action='store_false',
        dest='use_cosine_similarity',
        help='Use Euclidean distance instead of cosine similarity'
    )
    parser.add_argument(
        '--multi_layer_aggregation',
        action='store_true',
        default=True,
        help='Aggregate hidden states from multiple layers (default: True)'
    )
    parser.add_argument(
        '--no_multi_layer',
        action='store_false',
        dest='multi_layer_aggregation',
        help='Use only the last layer hidden states'
    )
    parser.add_argument(
        '--query_guided',
        action='store_true',
        default=True,
        help='Use query-guided token selection for k-means (default: True)'
    )
    parser.add_argument(
        '--no_query_guided',
        action='store_false',
        dest='query_guided',
        help='Disable query-guided selection'
    )
    parser.add_argument(
        '--query_weight',
        type=float,
        default=0.3,
        help='Weight for query relevance in selection (0-1). Higher = more query-focused. Default: 0.3'
    )
    
    args = parser.parse_args()
    main(args)