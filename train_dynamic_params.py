import argparse
import torch
import numpy as np
from datasets import load_dataset
from scipy.optimize import curve_fit
from bmrt import RecursiveCompressionEngine
from transformers import AutoTokenizer, AutoModelForCausalLM

def measure_nll_target(model, tokenizer, context, question, target, device):
    """
    Measures NLL(target | context, question).
    """
    # Construct input: [context] [question] [target]
    # We want loss only on [target].
    
    # Use chat template if appropriate or raw concatenation?
    # Model is Instruct, so ideally chat template.
    # But RecursiveCompressionEngine uses raw context + templated query.
    # Let's match that behavior.
    
    # 1. Tokenize context (already raw ids usually in engine, but here we do it again)
    # The engine computes compressed context.
    # If we are doing "Full" NLL, we use full context.
    # If we are doing "Compressed" NLL, we need the compressed KV from the engine.
    
    # Actually, to get NLL(target | compressed_context, question), we need to:
    # 1. Run engine to get compressed KV cache for (context + question).
    # 2. Append target tokens.
    # 3. Compute loss on target tokens given the KV cache.
    pass

def training_loop(args):
    print(f"Loading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We need a base engine to leverage its compression logic.
    # We'll re-instantiate or reset it for each sample/r.
    
    print(f"Loading dataset: {args.dataset_config}/{args.dataset_split}")
    ds = load_dataset("RMT-team/babilong", args.dataset_config, split=args.dataset_split)
    
    # Select a subset
    indices = range(args.num_samples)
    subset = [ds[i] for i in indices]
    
    results = [] # Stores (r, nll_c, y)
    
    # Pre-load model/tokenizer once to avoid overhead? 
    # The Engine loads it internally. To fit in memory, we might just instantiate Engine once 
    # and maybe hack it to change budget? Or re-instantiate.
    # Re-instantiating is safer but slower. 
    # Let's just instantiate once and update budget/clear cache.
    # But Engine computes budget derived things in __init__.
    # We'll use a helper to update budget.
    
    # We need one "Full" run per sample to get denominator NLL(full).
    # But wait, fitting requires valid range.
    # y = NLL(full) / NLL(compressed).
    
    # Retention ratios to test
    r_values = [float(x) for x in args.r_values.split(',')]
    
    print("Starting data collection...")
    
    # Initialize Engine with a dummy budget first (will update dynamically)
    engine = RecursiveCompressionEngine(
        model_path=args.model_path,
        selector_type=args.method,
        compression_mode='recursive', # Force recursive for now?
        backend='eager', # Safe default
        budget=4096, # Placeholder
        protection_divisor=4,
        block_size=args.block_size
    )
    
    for idx, sample in enumerate(subset):
        context = sample['input']
        question = sample['question']
        target = sample['target']
        
        print(f"Processing sample {idx+1}/{len(subset)}...")
        
        # 1. Measure NLL(c)
        context_ids = engine._tokenize(context)
        nll_c = engine._measure_nll(context_ids)
        print(f"  NLL(c) = {nll_c:.4f}")
        
        # 2. Measure NLL(full)
        # We can approximate "Full" by using r=1.0 (or just run model raw)
        # Let's run raw for ground truth.
        nll_full = get_target_nll_raw(engine, context, question, target)
        print(f"  NLL(full) = {nll_full:.4f}")
        
        # 3. Measure NLL(compressed) for each r
        for r in r_values:
            total_tokens = context_ids.shape[1]
            budget = int(r * total_tokens)
            if budget < 256: budget = 256 # Min floor
            
            # Update engine budget
            engine._configure_budget(budget)
            
            # Run compression
            # We need to capture the state BEFORE generation to compute NLL of target.
            # The engine.__call__ generates text.
            # We should probably expose a method `compress_and_score(context, question, target)`
            # Or manually call internal methods.
            
            try:
                nll_compressed = get_target_nll_compressed(engine, context, question, target)
                y = nll_full / nll_compressed
                print(f"    r={r:.2f} -> NLL(comp)={nll_compressed:.4f}, y={y:.4f}")
                
                results.append((r, nll_c, y))
            except Exception as e:
                print(f"    Error for r={r}: {e}")
                
        # Cleanup
        engine.cleanup()
        torch.cuda.empty_cache()

    print("\nData collection complete.")
    print(f"Collected {len(results)} points.")
    
    # Optimization
    print("Fitting parameters...")
    
    def func(X, alpha, beta):
        r, nll_c = X
        # b = alpha * nll_c + beta
        b = alpha * nll_c + beta
        tau = 0.95 # We are fitting f(r, c) = y? 
        # Wait, the paper says:
        # g(r, q, c) approx f(r, c)
        # We measure y = g(r, q, c).
        # We want to minimize (f(r, c) - y)^2.
        # f(r, c) formula:
        # f = (exp(rb - b) - exp(-b)) / (1 - exp(-b))
        
        # Vectorized for numpy
        # b can be array
        
        # Safe impl
        res = []
        for i in range(len(b)):
            bi = b[i]
            if np.abs(bi) < 1e-6:
                # Limit b->0 is f = r
                res.append(r[i])
                continue
                
            num = np.exp(r[i]*bi - bi) - np.exp(-bi)
            den = 1.0 - np.exp(-bi)
            val = num / den
            res.append(val)
        return np.array(res)

    # Unpack data
    R_data = np.array([x[0] for x in results])
    C_data = np.array([x[1] for x in results])
    Y_data = np.array([x[2] for x in results])
    
    # Initial guess
    p0 = [0.1, 0.1]
    
    try:
        popt, pcov = curve_fit(func, (R_data, C_data), Y_data, p0=p0, bounds=((-np.inf, -np.inf), (np.inf, np.inf)))
        alpha_opt, beta_opt = popt
        print(f"\nOptimization successful!")
        print(f"Alpha: {alpha_opt:.6f}")
        print(f"Beta:  {beta_opt:.6f}")
        
    except Exception as e:
        print(f"Optimization failed: {e}")

def get_target_nll_raw(engine, context, question, target):
    # Full inference NLL
    # Input: context + chat_template(question) + target
    model = engine.model
    tokenizer = engine.tokenizer
    
    # Encode
    msgs = [{"role": "user", "content": question}]
    query_tensor = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to(engine.device)
    context_tensor = engine._tokenize(context)
    target_tensor = engine._tokenize(" " + target) # Leading space?
    
    # Full sequence: [context] [query] [target]
    # We want loss on [target] only.
    
    input_ids = torch.cat([context_tensor, query_tensor, target_tensor], dim=1)
    
    # Labels: -100 everywhere except target
    labels = input_ids.clone()
    context_query_len = context_tensor.shape[1] + query_tensor.shape[1]
    labels[:, :context_query_len] = -100
    
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    
    return out.loss.item()

def get_target_nll_compressed(engine, context, question, target):
    # 1. Compress context + query
    # We use engine components but need to stop before generation
    
    context_ids = engine._tokenize(context)
    query_ids = engine._tokenize_query_with_chat_template(question)
    blocks = engine._split_into_blocks(context_ids)
    
    accumulated_kv_cache = None
    block_start_position = 0
    total_context_length = context_ids.shape[1]
    
    for i, block in enumerate(blocks):
        _, _, block_kv = engine._process_block(
            prefix_kv_cache=accumulated_kv_cache,
            block_ids=block,
            query_ids=query_ids,
            block_start_position=block_start_position,
            total_context_length=total_context_length
        )
        if engine.compression_mode == 'recursive':
            accumulated_kv_cache = block_kv
        else:
            from bmrt.utils import merge_kv_caches
            accumulated_kv_cache = merge_kv_caches(accumulated_kv_cache, block_kv)
        block_start_position += block.shape[1]

    # Handle final local tail if accumulate
    if engine.compression_mode == 'accumulate' and engine.prev_local_tail_kv is not None:
         from bmrt.utils import merge_kv_caches
         accumulated_kv_cache = merge_kv_caches(accumulated_kv_cache, engine.prev_local_tail_kv)
         
    # Now we have the compressed cache for (Context + Query).
    # We want to measure NLL of Target.
    target_tensor = engine._tokenize(" " + target)
    
    return measure_nll_with_cache(engine, target_tensor, accumulated_kv_cache, total_context_length + query_ids.shape[1])

def measure_nll_with_cache(engine, target_ids, past_key_values, start_pos):
    model = engine.model
    device = engine.device
    
    # We feed target_ids one by one or all at once?
    # If we feed all at once, we need position ids.
    
    seq_len = target_ids.shape[1]
    position_ids = torch.arange(start_pos, start_pos + seq_len, device=device).unsqueeze(0)
    
    # Prepare cache
    from transformers.cache_utils import DynamicCache
    if not isinstance(past_key_values, DynamicCache):
         past_key_values = DynamicCache.from_legacy_cache(past_key_values)
    
    # Logic:
    # forward(input_ids=target_ids, past_key_values=cache)
    # This computes logits for target_ids.
    # To compare with labels (target_ids), we usually shift inside the model loss,
    # BUT if we pass labels, HF model computes loss on input_ids against labels shifted.
    # If we provide past_key_values, the model assumes we are continuing.
    
    # If we pass labels=target_ids, the model will compute loss(generated=target_ids[i], label=target_ids[i+1]).
    # But it misses the first token prediction (output from query->target[0]).
    # To get exact NLL of target[0...N], we actually need the logits from the LAST token of the query...
    
    # Actually, simpler:
    # 1. Forward pass on target_ids with cache (produces logits for target[1..] and next)
    # 2. But we need logits for target[0]. Those come from the LAST step of context+query processing.
    #    The engine `_process_block` doesn't return logits.
    
    # Valid Approximation:
    # Just compute NLL of target[1...] given target[0]. Or assume target is long enough.
    # OR, we run one step on the last token of query (if we had it) to get first target token prob.
    # But we don't have the last query token in `target_ids`.
    
    # Let's adjust:
    # We can invoke model on target_ids.
    # We lose likelihood of first token. That's probably fine for optimization if consistent.
    # Or we can accept that `get_target_nll_raw` also does NLL over the whole sequence?
    # Actually `get_target_nll_raw` used labels with masking, so it includes target[0] prediction (from query[-1]).
    
    # To catch target[0], we need to re-run the last token of query?
    # That's expensive/complex with the cache state.
    # Let's just compute NLL on target_ids (which means P(target[1]|target[0]), etc.)
    # and use the same logic for `get_target_nll_raw`.
    
    # Modify `get_target_nll_raw` to also ignore first token of target?
    # Yes, to be apple-to-apples.
    
    # Wait, `get_target_nll_raw` computes loss on all tokens where label != -100.
    # If we set label for target[0] to target[0], it predicts target[0] from query[-1].
    
    # For compressed, we have the KV cache.
    # If we can't easily get logits for query[-1], we skip target[0].
    
    # Let's just proceed with NLL of target given cache.
    # We will ignore the first token's probability to keep it simple.
    # We must ensure `get_target_nll_raw` DOES THE SAME.
    
    with torch.no_grad():
        outputs = model(input_ids=target_ids, position_ids=position_ids, past_key_values=past_key_values)
        logits = outputs.logits # [1, seq_len, vocab]
        
    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    return loss.item()

def get_target_nll_raw_compatible(engine, context, question, target):
    # Same as above but ignore target[0]
    model = engine.model
    tokenizer = engine.tokenizer
    
    msgs = [{"role": "user", "content": question}]
    query_tensor = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt').to(engine.device)
    context_tensor = engine._tokenize(context)
    target_tensor = engine._tokenize(" " + target)
    
    input_ids = torch.cat([context_tensor, query_tensor, target_tensor], dim=1)
    
    labels = input_ids.clone()
    # Mask everything up to target start + 1 (ignore predict target[0])
    prefix_len = context_tensor.shape[1] + query_tensor.shape[1]
    labels[:, :prefix_len + 1] = -100 # Mask context, query, and target[0]
    
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
        
    return out.loss.item()


# Patch the functions into the global scope or call them carefully
# `get_target_nll_raw` -> `get_target_nll_raw_compatible`
get_target_nll_raw = get_target_nll_raw_compatible

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_config", default="16k")
    parser.add_argument("--dataset_split", default="qa1")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--r_values", default="0.1,0.25,0.5,0.75,0.9")
    parser.add_argument("--method", default="exact")
    parser.add_argument("--block_size", type=int, default=1024)
    
    args = parser.parse_args()
    training_loop(args)
