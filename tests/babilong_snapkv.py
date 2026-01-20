import argparse
import os
import time
import torch
import gc
import datetime
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from kvpress import SnapKVPress

def main(args):
    print("\n" + "="*60)
    print("  SnapKV Babilong Inference Multi-Sample Test")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Model: {args.model_path}")
    print(f"  - Method: SnapKV")
    print(f"  - Compression Ratio: {args.compression_ratio}")
    print(f"  - Max New Tokens: {args.max_new_tokens}")
    print(f"  - Dataset: {args.dataset_config} ({args.dataset_split})")
    print("="*60)

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data ---")
    dataset_name = "RMT-team/babilong"
    try:
        ds = load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # --- 2. Initialize Model & Pipeline ---
    print(f"\n--- 2. Loading Model {args.model_path} ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        # Use a specific device and enable Flash Attention via model_kwargs
        device = "cuda:0"
        model_kwargs = {"attn_implementation": "flash_attention_2"}
        pipe = pipeline(
            "kv-press-text-generation",
            model=args.model_path,
            device=device,
            model_kwargs=model_kwargs,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        press = SnapKVPress(compression_ratio=args.compression_ratio)
        print("Model and SnapKV pipeline loaded successfully.")
    except Exception as e:
        print(f"Failed to load model/pipeline: {e}")
        return

    # --- 3. Run Inference ---
    print("\n--- 3. Running Inference ---")
    correct = 0
    total = 0
    end_idx = min(args.start_sample_index + args.num_samples, len(ds))
    
    total_samples = end_idx - args.start_sample_index
    print(f"Processing {total_samples} samples...")

    for idx, i in enumerate(range(args.start_sample_index, end_idx)):
        # Manual Progress Bar
        sys.stdout.write(f"\rProgress: [{idx+1}/{total_samples}] ({(idx+1)/total_samples*100:.1f}%)")
        sys.stdout.flush()
        
        total += 1
        sample = ds[i]
        context = sample['input']
        query = sample['question']
        target = sample['target']

        # Construct prompt
        # Using a simple format suitable for instruction tuned models or base completion
        # Ideally, we should use the model's chat template, but here we'll stick to simple concatenation
        # to match general practice unless a specific template is required.
        # However, pred_snapkv uses a template. For Babilong, usually:
        # User: {context}\n\n{query}
        prompt = f"{context}\n\n{query}"

        # Context Truncation (Optional but good for safety)
        # Check against a large max length like 120k or model max
        # For simplicity, we rely on SnapKV to handle the compression, 
        # but explicit truncation prevents OOM on raw input processing if needed.
        # We'll skip explicit truncation here to let SnapKV handle it unless it fails.

        try:
            start = time.time()
            # Run inference
            output = pipe(
                prompt,
                press=press,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                return_full_text=False
            )
            duration = time.time() - start

            # Extract prediction
            if isinstance(output, dict) and "answer" in output:
                prediction = output["answer"]
            elif isinstance(output, list) and "generated_text" in output[0]:
                prediction = output[0]["generated_text"]
            else:
                prediction = str(output)
            
            # Clean up potential prompt leakage
            if prediction.startswith(prompt):
                 prediction = prediction[len(prompt):]

            # Evaluate
            if target.lower() in prediction.lower() or prediction.lower() in target.lower():
                correct += 1
                match_status = "✓ PASS"
            else:
                match_status = "✗ FAIL"
            
            # Show output
            accuracy_so_far = correct / total * 100
            print(f"\n  Sample {i}: {match_status}")
            print(f"  Target: {target}")
            print(f"  Prediction: {prediction.strip()[:200]}{'...' if len(prediction.strip()) > 200 else ''}")
            print(f"  Running Accuracy: {accuracy_so_far:.2f}% ({correct}/{total})")

        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    print("") # Newline
    if total > 0:
        accuracy = correct/total*100
        print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total})")

        # Append to results file
        results_line = (
            f"{datetime.datetime.utcnow().isoformat()} | model={args.model_path} | "
            f"dataset={args.dataset_config}/{args.dataset_split} | method=snapkv | "
            f"compression_ratio={args.compression_ratio} | "
            f"samples=start:{args.start_sample_index},n:{args.num_samples} | "
            f"correct={correct}/{total} | accuracy={accuracy:.2f}%\n"
        )

        try:
            with open(args.results_file, 'a', encoding='utf-8') as rf:
                rf.write(results_line)
        except Exception as e:
            print(f"Failed to write results to {args.results_file}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='gradientai/Llama-3-8B-Instruct-Gradient-1048k')
    parser.add_argument('--dataset_config', default='128k')
    parser.add_argument('--dataset_split', default='qa2')
    parser.add_argument('--start_sample_index', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--compression_ratio', type=float, default=0.125)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--results_file', default='accuracies_snapkv.txt')
    
    args = parser.parse_args()
    main(args)
