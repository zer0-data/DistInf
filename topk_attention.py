# topk_attention.py
# Parallel Smart Summary with Query-Guided BM25/TF-IDF Selection
# Optimized for Flash Attention 2 by creating dense, position-aware summary tensors.

from typing import Optional, Tuple, List, Dict
import math
import re
from collections import Counter
import torch
from transformers.cache_utils import DynamicCache

# =============================================================================
# SCORING UTILITIES (BM25 + TF-IDF)
# =============================================================================

class TextScorer:
    """
    Implements the Hybrid BM25 + Global IDF scoring logic.
    """
    def __init__(self, global_text: str):
        self.idf_map = self._compute_global_idf(global_text)
        
    def _simple_tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def _compute_global_idf(self, text: str) -> Dict[str, float]:
        """Pre-compute IDF scores for the entire context."""
        words = self._simple_tokenize(text)
        total_words = len(words)
        word_counts = Counter(words)
        # IDF = log(Total / (Freq + 1))
        return {w: math.log(total_words / (c + 1)) for w, c in word_counts.items()}

    def score_sentence(self, sentence: str, query_tokens: List[str]) -> float:
        """
        Calculate Score = (Query_BM25 * Alpha) + (Context_TFIDF * Beta)
        """
        sent_words = self._simple_tokenize(sentence)
        if not sent_words:
            return 0.0
        
        # 1. Query Relevance (BM25-style)
        query_score = 0.0
        for q_word in query_tokens:
            if q_word in sent_words:
                tf = sent_words.count(q_word)
                # Saturation: tf / (tf + 1.5) * IDF
                idf = self.idf_map.get(q_word, 0)
                query_score += idf * (tf / (tf + 1.5))
        
        # 2. Context Importance (TF-IDF Density)
        context_score = sum(self.idf_map.get(w, 0) for w in sent_words) / len(sent_words)
        
        # Hybrid Mix: Prioritize Query heavily
        return (query_score * 10.0) + context_score


# =============================================================================
# PARALLEL SMART PROCESSOR (Flash Attention Optimized)
# =============================================================================

class ParallelSmartSummaryProcessor:
    """
    Processor that selects summaries using BM25/TF-IDF and constructs a dense KV cache 
    compatible with Flash Attention 2.
    """
    
    def __init__(
        self,
        model_path: str,
        top_k: int = 256,
        block_size: int = 2048,
        max_new_tokens: int = 100,
        stop_words: Optional[List[str]] = None,
        anchor_size: int = 64,
        local_window_size: int = 64,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.model_path = model_path
        self.top_k = top_k
        self.block_size = block_size
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words or []
        self.anchor_size = anchor_size
        self.local_window_size = local_window_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"\n[ParallelSmartProcessor] Initializing...")
        print(f"  Model: {model_path}")
        print(f"  Strategy: Anchor({anchor_size}) + Local({local_window_size}) + BM25-TopK({top_k})")
        print(f"  Attention: Flash Attention 2 (Enabled)")
        
        # Load tokenizer (Fast tokenizer required for offsets)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("  Loading model with flash_attention_2...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            torch_dtype=torch.bfloat16, # FA2 requires bf16 or fp16
            trust_remote_code=True,
            attn_implementation='flash_attention_2', 
        )
        self.model.eval()

    def _tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer.encode(
            text, return_tensors='pt', add_special_tokens=False
        ).to(self.device)

    def _tokenize_query_with_chat_template(self, query_text: str) -> torch.Tensor:
        """Tokenize query using chat template for consistency with generation."""
        messages = [{"role": "user", "content": query_text}]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)
        else:
            return self._tokenize(query_text)

    def _split_into_blocks(self, token_ids: torch.Tensor) -> List[torch.Tensor]:
        return list(token_ids.split(self.block_size, dim=1))

    def _get_sentence_intervals(self, text: str) -> List[Tuple[int, int]]:
        pattern = re.compile(r'(.*?[.?!])(?:\s|$)')
        matches = list(pattern.finditer(text))
        intervals = []
        if not matches and len(text) > 0:
             return [(0, len(text))]
        for m in matches:
            intervals.append(m.span(1))
        return intervals

    def _sample_bm25_from_block(
        self,
        block_ids: torch.Tensor,
        scorer: TextScorer,
        query_tokens: List[str],
        block_start_position: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects tokens from the block based on sentence importance (BM25/TF-IDF).
        Returns dense tensor of selected tokens and their original position IDs.
        """
        block_len = block_ids.shape[1]
        
        # 1. Identify Fixed Indices (Anchor + Local)
        anchor_end = min(self.anchor_size, block_len)
        local_start = max(anchor_end, block_len - self.local_window_size)
        fixed_indices = set(range(anchor_end)) | set(range(local_start, block_len))
        
        # 2. Decode and Map
        block_text = self.tokenizer.decode(block_ids[0], skip_special_tokens=False)
        enc = self.tokenizer(block_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc['offset_mapping']
        
        if len(offsets) != block_len:
            indices = sorted(list(fixed_indices))
            indices_t = torch.tensor(indices, device=self.device, dtype=torch.long)
            return block_ids[:, indices_t], (block_start_position + indices_t).unsqueeze(0)

        # 3. Sentence Scoring
        sentence_spans = self._get_sentence_intervals(block_text)
        scored_sentences = []
        
        for start_char, end_char in sentence_spans:
            sent_text = block_text[start_char:end_char]
            score = scorer.score_sentence(sent_text, query_tokens)
            
            sent_token_indices = []
            for i, (t_start, t_end) in enumerate(offsets):
                t_center = (t_start + t_end) / 2
                if start_char <= t_center < end_char:
                    sent_token_indices.append(i)
            
            if sent_token_indices:
                unique_tokens = [t for t in sent_token_indices if t not in fixed_indices]
                if unique_tokens:
                    scored_sentences.append((score, unique_tokens))

        # 4. Selection
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        selected_middle_indices = []
        current_tokens = 0
        for score, tokens in scored_sentences:
            if current_tokens + len(tokens) > self.top_k:
                break
            selected_middle_indices.extend(tokens)
            current_tokens += len(tokens)
            
        # 5. Final Dense Construction
        final_indices_set = fixed_indices | set(selected_middle_indices)
        final_indices = sorted(list(final_indices_set))
        
        indices_tensor = torch.tensor(final_indices, device=self.device, dtype=torch.long)
        
        summary_ids = block_ids.index_select(dim=1, index=indices_tensor)
        summary_positions = (block_start_position + indices_tensor).unsqueeze(0)
        
        return summary_ids, summary_positions

    @torch.no_grad()
    def _build_kv_cache_from_summaries(
        self,
        summaries: List[torch.Tensor],
        summary_original_positions: List[torch.Tensor],
    ) -> Tuple:
        all_summary_token_ids = torch.cat(summaries, dim=1)
        all_summary_positions = torch.cat(summary_original_positions, dim=1)
        
        outputs = self.model(
            input_ids=all_summary_token_ids,
            position_ids=all_summary_positions,
            use_cache=True,
            output_attentions=False,
        )
        return outputs.past_key_values, all_summary_token_ids, all_summary_positions

    @torch.no_grad()
    def _generate(
        self,
        query_ids: torch.Tensor,
        kv_cache: Tuple,
        original_context_len: int # FIX: Explicitly pass the original context length
    ) -> torch.Tensor:
        """
        Phase 3: Generate response by projecting query onto sparse KV cache.
        """
        # FIX: The query must start AFTER the original context, not after the compressed cache.
        # This ensures RoPE distances are correct relative to the original document structure.
        start_pos = original_context_len
        
        position_ids = torch.arange(
            start_pos, start_pos + query_ids.shape[1], device=self.device
        ).unsqueeze(0)
        
        past_key_values = DynamicCache.from_legacy_cache(kv_cache)
        
        outputs = self.model(
            input_ids=query_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        current_cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        generated_tokens = [next_token]
        
        for _ in range(self.max_new_tokens - 1):
            outputs = self.model(
                input_ids=next_token,
                past_key_values=current_cache,
                use_cache=True,
            )
            
            current_cache = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated_tokens.append(next_token)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return torch.cat(generated_tokens, dim=1)

    def _get_output_text(self, token_ids: torch.Tensor) -> str:
        text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        for stop_word in self.stop_words:
            text = text.split(stop_word)[0]
        return text.strip()

    def __call__(self, prompt_context: str, prompt_query: str) -> Dict[str, List[str]]:
        print("=" * 60)
        print("Parallel BM25 Processing Pipeline (Flash Attention Enabled)")
        print("=" * 60)
        
        scorer = TextScorer(prompt_context)
        query_tokens = scorer._simple_tokenize(prompt_query)
        
        context_ids = self._tokenize(prompt_context)
        query_ids = self._tokenize_query_with_chat_template(prompt_query)
        
        # Calculate original context length for correct position ID generation later
        original_context_len = context_ids.shape[1]

        print(f"\nContext length: {original_context_len} tokens")
        print(f"Query length: {query_ids.shape[1]} tokens")
        
        blocks = self._split_into_blocks(context_ids)
        
        summaries = []
        summary_positions = []
        block_start_pos = 0
        
        print("\n--- Phase 1: Parallel Selection & Slicing ---")
        for i, block in enumerate(blocks):
            s_ids, s_pos = self._sample_bm25_from_block(
                block, scorer, query_tokens, block_start_pos
            )
            summaries.append(s_ids)
            summary_positions.append(s_pos)
            block_start_pos += block.shape[1]
            
        print("\n--- Phase 2: Flash Attention Cache Prefill ---")
        kv_cache, _, _ = self._build_kv_cache_from_summaries(
            summaries, summary_positions
        )
        
        total_cache_len = kv_cache[0][0].shape[2]
        print(f"  Final Cache Size: {total_cache_len} tokens (Original: {original_context_len})")
        print(f"  Compression Ratio: {original_context_len / total_cache_len:.1f}x")
        
        del blocks, summaries, summary_positions, context_ids
        torch.cuda.empty_cache()
        
        print("\n--- Phase 3: Generation ---")
        # Pass the original length so query positions are correct
        generated_ids = self._generate(query_ids, kv_cache, original_context_len)
        output_text = self._get_output_text(generated_ids)
        print(f"\nGenerated {generated_ids.shape[1]} tokens")
        print("=" * 60)
        
        return {'text': [output_text]}