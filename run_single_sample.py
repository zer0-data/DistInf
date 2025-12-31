import argparse
import os
import time
import torch
import gc
import datetime
from datasets import load_dataset

from bmrt import RecursiveCompressionEngine

def main(args):
    print("\n" + "="*60)
    print("  Recursive BMRT Inference Multi-Sample Test (v2.0)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  - Model: {args.model_path}")
    print(f"  - Method: {args.method} (Backend: {args.backend})")
    if args.method == 'lsh' or args.method == 'hybrid':
        print(f"  - LSH Mode: {args.lsh_mode}")
        print(f"  - LSH Config: num_bits={args.num_bits}, num_tables={args.num_tables}")
    print(f"  - Budget: {args.budget}")
    print(f"  - Protection Divisor: {args.protection_divisor}")
    print(f"  - Block size: {args.block_size}")
    print(f"  - Top-K: {args.top_k} (Legacy arg, now implicitly global budget)")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print("="*60)

    # --- 1. Load Data ---
    print("\n--- 1. Loading Data ---")
    dataset_name = "RMT-team/babilong"
    try:
        ds = load_dataset(dataset_name, args.dataset_config, split=args.dataset_split)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Helper to build engine per-sample (so we can fully release memory afterwards)
    def build_engine():
        print("\n--- Loading RecursiveCompressionEngine ---")
        try:
            eng = RecursiveCompressionEngine(
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
                num_bits=args.num_bits,
                num_tables=args.num_tables,
                hybrid_primary=args.hybrid_primary,
                hybrid_secondary=args.hybrid_secondary,
                hybrid_ratio=args.hybrid_ratio,
            )
            print("Engine loaded successfully.")
            return eng
        except Exception as e:
            print(f"Failed to load engine: {e}")
            return None

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

        engine = build_engine()
        if engine is None:
            print("Skipping sample due to engine load failure.")
            continue

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

        # Thorough cleanup after each sample to free model and KV caches
        try:
            engine.cleanup()
        except Exception:
            pass
        try:
            del engine
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    if total > 0:
        accuracy = correct/total*100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{total})")

        # Append a results line to the results file (if provided)
        results_line = (
            f"{datetime.datetime.utcnow().isoformat()} | model={args.model_path} | "
            f"dataset={args.dataset_config}/{args.dataset_split} | method={args.method} | "
            f"lsh_mode={args.lsh_mode} | hybrid={args.hybrid_primary}+{args.hybrid_secondary}:{args.hybrid_ratio} | "
            f"backend={args.backend} | compression_mode={args.compression_mode} | "
            f"sampling_config=budget:{args.budget},block_size:{args.block_size},top_k:{args.top_k} | "
            f"protection_divisor={args.protection_divisor} | "
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
    parser.add_argument('--model_path', default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--method', default='exact', choices=['exact', 'lsh', 'hybrid'])
    parser.add_argument('--lsh_mode', default='frequency_rank', choices=['frequency_rank', 'magicpig_baseline'])
    parser.add_argument('--compression_mode', default='accumulate', choices=['accumulate', 'recursive'])
    parser.add_argument('--backend', default='eager', choices=['eager', 'flash'])
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--protection_divisor', type=int, default=4)
    parser.add_argument('--block_size', type=int, default=4096)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--stop_words', default='')
    
    # LSH args
    parser.add_argument('--num_bits', type=int, default=6, help='Number of bits per LSH hash')
    parser.add_argument('--num_tables', type=int, default=4, help='Number of LSH hash tables')
    
    # Hybrid args
    parser.add_argument('--hybrid_primary', default='exact')
    parser.add_argument('--hybrid_secondary', default='lsh')
    parser.add_argument('--hybrid_ratio', type=float, default=0.5)
    
    # Dataset args
    parser.add_argument('--dataset_config', default='16k')
    parser.add_argument('--dataset_split', default='qa1')
    parser.add_argument('--start_sample_index', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=0, help="Unused but kept for compatibility")
    parser.add_argument('--results_file', default='accuracies.txt', help='File to append accuracy results')
    
    args = parser.parse_args()
    main(args)
