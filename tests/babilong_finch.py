import argparse
import torch
from datasets import load_dataset
from transformers import pipeline
from kvpress import FinchPress
import os

def evaluate_babilong(context_size, compression_ratio, chunk_size, output_file="results.txt"):
    # 1. Setup Device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} with Context: {context_size}, Ratio: {compression_ratio}, Chunk: {chunk_size}")

    # 2. Load Dataset
    # Loading the specific split based on context_size argument (e.g., "16k", "32k")
    print(f"Loading RMT-team/babilong ({context_size})...")
    dataset = load_dataset("RMT-team/babilong", context_size, split="qa1")

    # 3. Setup Pipeline
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Initialize the custom kv-press pipeline
    pipe = pipeline(
        "kv-press-text-generation", 
        model=model_id, 
        device=device, 
        model_kwargs={"attn_implementation": "flash_attention_2"},
        torch_dtype=torch.float16
    )

    # 4. Initialize FinchPress
    press = FinchPress(
        compression_ratio=float(compression_ratio),
        chunk_size=int(chunk_size)
    )

    correct_count = 0
    total_count = 0
    
    print("Starting evaluation...")
    
    for i, item in enumerate(dataset):
        context = item['input']
        question = item['question']
        ground_truth = item['target']
        
        try:
            result = pipe(
                context, 
                question=question, 
                press=press,
                max_new_tokens=50,
                do_sample=False # Deterministic for evaluation
            )
            
            prediction = result["answer"].strip()
            
            if ground_truth.lower() in prediction.lower():
                correct_count += 1
            
            total_count += 1
            
            # Optional: Print progress every 10 samples
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} samples. Current Accuracy: {correct_count/total_count:.2%}")

        except Exception as e:
            print(f"Error on sample {i}: {e}")
            continue

    # 6. Calculate Final Accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2%}")

    # 7. Save Results to Text File
    result_line = (
        f"Context Size: {context_size} | "
        f"Compression Ratio: {compression_ratio} | "
        f"Chunk Size: {chunk_size} | "
        f"Accuracy: {accuracy:.4f}\n"
    )
    
    with open(output_file, "a") as f:
        f.write(result_line)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FinchPress on Babilong")
    
    # Arguments for the 3 parameters
    parser.add_argument("--context_size", type=str, required=True, help="Context size (e.g., '16k', '32k')")
    parser.add_argument("--compression_ratio", type=float, required=True, help="Compression ratio (e.g., 0.125)")
    parser.add_argument("--chunk_size", type=int, required=True, help="Chunk size (e.g., 256)")
    
    args = parser.parse_args()
    
    evaluate_babilong(args.context_size, args.compression_ratio, args.chunk_size)
