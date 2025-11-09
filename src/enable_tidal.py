def enable_tidal(
    model,
    attn_type="tidal",
    top_k=256,
    sparse_layer_start=2,
    correction_layer=13,
    attention_sink=0,
    lim_ratio=1,
    **kwargs,  # <-- NEW: Accept arbitrary keyword arguments
):
    if attn_type == "lim" or attn_type == "tidal": # <-- MODIFIED: Handle 'tidal'
        print(f"Tidal/LIM Enabled: attention_sink: {attention_sink}")
        print(f"token budget: {top_k}")
        print(f"sparse layer starts from: Layer {sparse_layer_start}")
        print(f"reselection layer: {correction_layer}")
        
        # NEW: Extract and print selection_layers if present
        selection_layers = kwargs.get("selection_layers", [])
        if selection_layers:
            print(f"prefill selection layers: {selection_layers}")
            
        model_type = model.config.model_type

        if "llama" in model_type:
            # The import path assumes 'enable_tidal.py' is in 'src'
            # and 'modify_llama_lim.py' is in 'src/tidal_build'
            from .tidal_build.modify_llama_lim import (
                enable_llama_tidal_attention,
            )

            enable_llama_tidal_attention(
                model,
                top_k,
                attn_type,
                sparse_layer_start,
                correction_layer,
                attention_sink,
                lim_ratio,
                **kwargs,  # <-- NEW: Pass all kwargs down
            )
    return