# DistInf
A distributed inference framework supporting Star Attention and custom accuracy models for long-context language model inference.

## Using Top-K summarization
python run_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --block_size 4096 \
    --k_summary_size 128 \
    --max_new_tokens 100 \
    --summary_method top_k

## Using K-means summarization
python run_single_sample.py \
    --model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --block_size 4096 \
    --k_summary_size 128 \
    --max_new_tokens 100 \
    --summary_method kmeans