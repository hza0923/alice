"""Legacy profile component name -> scheduler module id (matches legacy scheduler)."""

PROFILE_TO_MODULE: dict[str, str] = {
    "embed_tokens": "input_embed",
    "qkv_rope": "qkv_projection",
    "attn_qk": "attn_qk",
    "attn_av": "attn_av",
    "attn_wo": "o_projection",
    "up_proj": "up_projection",
    "down_proj": "down_projection",
    "lm_head": "lm_head",
}

KV_MODULES = frozenset({"attn_qk", "attn_av"})

# lm_head is treated as fixed latency; prefill registry uses decode `average` ms (see fit_prefill_time).
LM_HEAD_MODULE = "lm_head"
