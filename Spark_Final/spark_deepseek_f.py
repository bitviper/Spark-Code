# spark_deepseek_f.py
# True MLA (Multi‑Head Latent Attention) implementation.
# Features:
# - Decoupled RoPE (content compressed, positional added later)
# - Correct decompression shapes (w_uk, w_uv map to head dimensions)
# - Sliced RoPE for training/generation consistency
# - Caching of k_rope along with latent content vectors
# - Flash Attention via xla implementation

import jax
import jax.numpy as jnp
from functools import partial
from jax import remat

from spark_llm_f import (
    CONFIG as LLM_CONFIG,
    apply_rope,
    rms_norm,
    mlp,
    init_optimizer_state,
    apply_adamw_update,
    clip_grads,
    save_checkpoint,
    load_checkpoint,
    precompute_rope_freqs,
)

USE_FP16 = False

CONFIG = LLM_CONFIG.copy()
CONFIG.update({
    "latent_dim": 16,                
})

def cast_to_fp16(pytree):
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x, pytree
    )

# -------------------------------------------------------------------
# MLA Attention (Training)
# -------------------------------------------------------------------
def attention_layer(x, layer, num_heads, num_kv_heads, head_dim, cos, sin, mask):
    B, T, D = x.shape
    rope_dim = head_dim // 2
    content_dim = head_dim - rope_dim

    # 1. Q Projection + Full RoPE
    q = (x @ layer["self_attn"]["q_proj"]).reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    q = apply_rope(q, cos, sin)

    # 2. MLA Content Compression
    c_k = x @ layer["mla"]["w_ck"]  # (B, T, latent_dim)
    c_v = x @ layer["mla"]["w_cv"]  # (B, T, latent_dim)

    # 3. Decoupled RoPE Projection (sliced RoPE)
    k_rope = x @ layer["mla"]["w_rope"]  # (B, T, rope_dim)
    k_rope = apply_rope(k_rope[:, None, :, :], cos[..., :rope_dim], sin[..., :rope_dim]).squeeze(1)

    # 4. Decompress Content (shapes now correct)
    k_content = (c_k @ layer["mla"]["w_uk"]).reshape(B, T, num_kv_heads, content_dim).transpose(0, 2, 1, 3)
    v_full = (c_v @ layer["mla"]["w_uv"]).reshape(B, T, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    # 5. Split Rope part into KV heads and Concatenate
    # FIX: Use broadcast_to instead of reshape to match num_kv_heads dimension
    k_rope = jnp.broadcast_to(k_rope[:, :, None, :], (B, T, num_kv_heads, rope_dim)).transpose(0, 2, 1, 3)
    k_full = jnp.concatenate([k_content, k_rope], axis=-1)

    # GQA broadcasting
    k_rep = jnp.repeat(k_full, num_heads // num_kv_heads, axis=1)
    v_rep = jnp.repeat(v_full, num_heads // num_kv_heads, axis=1)

    # Flash Attention
    # FIX: Pass boolean mask to `mask` argument, not `bias`
    out = jax.nn.dot_product_attention(
        query=q.transpose(0, 2, 1, 3), 
        key=k_rep.transpose(0, 2, 1, 3), 
        value=v_rep.transpose(0, 2, 1, 3),
        mask=mask, is_causal=False, implementation='xla'
    )
    out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ layer["self_attn"]["o_proj"]

@partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable)
def transformer_layer_fn(layer, x, cos, sin, mask):
    res = x
    x = rms_norm(x, layer["rms_attn"]["scale"], CONFIG["eps_rms"])
    x = res + attention_layer(x, layer, CONFIG["num_heads"], CONFIG["num_kv_heads"],
                              CONFIG["d_model"] // CONFIG["num_heads"],
                              cos, sin, mask)
    res = x
    x = rms_norm(x, layer["rms_ffn"]["scale"], CONFIG["eps_rms"])
    x = res + mlp(x, layer)
    return x

# -------------------------------------------------------------------
# Forward pass
# -------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2,3,4,5,6))
def forward(params, input_ids, seq_len, num_heads, num_kv_heads, head_dim, eps_rms, cos, sin):
    if USE_FP16:
        params = cast_to_fp16(params)
    B, T = input_ids.shape
    tok_emb = params["tok_embeddings"]["weight"]
    x = tok_emb[input_ids] * jnp.sqrt(CONFIG["d_model"])

    mask = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))
    for layer in params["transformer"]["layers"]:
        x = transformer_layer_fn(layer, x, cos[:, :, :T, :], sin[:, :, :T, :], mask)

    x = rms_norm(x, params["transformer"]["rms_final"]["scale"], eps_rms)
    return (x @ tok_emb.T).astype(jnp.float32)

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train_step_mla(params, opt_state, key, ids, cos, sin):
    B, T = CONFIG["batch_size"], CONFIG["seq_len"]
    key, sub = jax.random.split(key)
    idx = jax.random.randint(sub, (B,), 0, ids.shape[0] - T - 1)
    x = jax.vmap(lambda s: jax.lax.dynamic_slice(ids, (s,), (T,)))(idx)
    y = jax.vmap(lambda s: jax.lax.dynamic_slice(ids, (s+1,), (T,)))(idx)

    def loss_fn(p):
        if USE_FP16: p = cast_to_fp16(p)
        logits = forward(p, x, T, CONFIG["num_heads"], CONFIG["num_kv_heads"],
                         CONFIG["d_model"] // CONFIG["num_heads"], CONFIG["eps_rms"], cos, sin)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.sum(target * log_probs, axis=-1))

    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads, grad_norm = clip_grads(grads, CONFIG["max_grad_norm"])
    new_params, new_opt_state = apply_adamw_update(
        params, grads, opt_state, CONFIG["lr"], CONFIG["adam_b1"],
        CONFIG["adam_b2"], CONFIG["adam_eps"], CONFIG["weight_decay"]
    )
    return new_params, new_opt_state, loss, grad_norm, key

@partial(jax.jit, static_argnums=(3,))
def train_chunk_mla(params, opt_state, key, chunk_size, ids):
    head_dim = CONFIG["d_model"] // CONFIG["num_heads"]
    cos, sin = precompute_rope_freqs(CONFIG["seq_len"], head_dim)
    
    def body(carry, _):
        p, o, k = carry
        p, o, loss, grad_norm, k = train_step_mla(p, o, k, ids, cos, sin)
        return (p, o, k), (loss, grad_norm)
        
    return jax.lax.scan(body, (params, opt_state, key), None, length=chunk_size)

# -------------------------------------------------------------------
# Generation (Caching k_rope and latent content)
# -------------------------------------------------------------------
@partial(jax.jit, static_argnums=(3,4,5,6,7))
def generation_step_cached(params, kv_cache, step_idx, input_ids,
                           seq_len, num_heads, num_kv_heads, head_dim, eps_rms,
                           cos_full, sin_full):
    if USE_FP16:
        params = cast_to_fp16(params)
    B, T = input_ids.shape
    rope_dim = head_dim // 2
    content_dim = head_dim - rope_dim
    
    tok_emb = params["tok_embeddings"]["weight"]
    x = tok_emb[input_ids] * jnp.sqrt(CONFIG["d_model"])

    # Slice RoPE: full head_dim for Q, rope_dim for k_rope
    cos_q = jax.lax.dynamic_slice(cos_full, (0, 0, step_idx, 0), (1, 1, 1, head_dim))
    sin_q = jax.lax.dynamic_slice(sin_full, (0, 0, step_idx, 0), (1, 1, 1, head_dim))
    cos_kr = jax.lax.dynamic_slice(cos_full, (0, 0, step_idx, 0), (1, 1, 1, rope_dim))
    sin_kr = jax.lax.dynamic_slice(sin_full, (0, 0, step_idx, 0), (1, 1, 1, rope_dim))

    new_ck_cache, new_cv_cache, new_kr_cache = [], [], []

    for i, layer in enumerate(params["transformer"]["layers"]):
        norm_x = rms_norm(x, layer["rms_attn"]["scale"], eps_rms)

        # 1. Q Projection + Full RoPE
        q = (norm_x @ layer["self_attn"]["q_proj"]).reshape(B, 1, num_heads, head_dim).transpose(0, 2, 1, 3)
        q = apply_rope(q, cos_q, sin_q)

        # 2. Compress Content (preserve 3D shape)
        c_k = norm_x @ layer["mla"]["w_ck"]  # (B, 1, latent_dim)
        c_v = norm_x @ layer["mla"]["w_cv"]

        # 3. k_rope with RoPE, then squeeze back to 3D
        k_rope = apply_rope(
            (norm_x @ layer["mla"]["w_rope"])[:, None, :, :], cos_kr, sin_kr
        ).squeeze(1)  # (B, 1, rope_dim)

        # Update caches
        layer_ck = jax.lax.dynamic_update_slice(kv_cache["c_k"][i], c_k, (0, step_idx, 0))
        layer_cv = jax.lax.dynamic_update_slice(kv_cache["c_v"][i], c_v, (0, step_idx, 0))
        layer_kr = jax.lax.dynamic_update_slice(kv_cache["k_rope"][i], k_rope, (0, step_idx, 0))
        
        new_ck_cache.append(layer_ck)
        new_cv_cache.append(layer_cv)
        new_kr_cache.append(layer_kr)

        # 4. Decompress & split into heads
        k_content = (layer_ck @ layer["mla"]["w_uk"]).reshape(1, seq_len, num_kv_heads, content_dim).transpose(0, 2, 1, 3)
        v_full = (layer_cv @ layer["mla"]["w_uv"]).reshape(1, seq_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        
        # FIX: Use broadcast_to instead of reshape for k_rope_full
        k_rope_full = jnp.broadcast_to(layer_kr[:, :, None, :], (1, seq_len, num_kv_heads, rope_dim)).transpose(0, 2, 1, 3)
        
        k_full = jnp.concatenate([k_content, k_rope_full], axis=-1)

        k_rep = jnp.repeat(k_full, num_heads // num_kv_heads, axis=1)
        v_rep = jnp.repeat(v_full, num_heads // num_kv_heads, axis=1)

        # FIX: Replaced dynamic_slice mask logic with simple broadcasted boolean array
        active_mask = (jnp.arange(seq_len) <= step_idx)[None, None, None, :]
        
        # FIX: Pass `active_mask` to mask, not bias
        out = jax.nn.dot_product_attention(
            query=q.transpose(0, 2, 1, 3), key=k_rep.transpose(0, 2, 1, 3), 
            value=v_rep.transpose(0, 2, 1, 3), mask=active_mask, is_causal=False, implementation='xla'
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, CONFIG["d_model"])

        x = x + out @ layer["self_attn"]["o_proj"]
        x = x + mlp(rms_norm(x, layer["rms_ffn"]["scale"], eps_rms), layer)

    x = rms_norm(x, params["transformer"]["rms_final"]["scale"], eps_rms)
    logits = x @ tok_emb.T

    return logits[:, -1, :], {
        "c_k": jnp.stack(new_ck_cache), 
        "c_v": jnp.stack(new_cv_cache),
        "k_rope": jnp.stack(new_kr_cache)
    }

def generate_text(params, sp, prompt_text, length=80, temperature=0.95,
                  top_k=40, top_p=0.85, repetition_penalty=1.2):
    seq_len = CONFIG["seq_len"]
    head_dim = CONFIG["d_model"] // CONFIG["num_heads"]
    num_layers = CONFIG["num_layers"]
    num_kv_heads = CONFIG["num_kv_heads"]
    rope_dim = head_dim // 2

    input_ids = sp.encode(prompt_text)[:seq_len - 1]
    cur_ids = list(input_ids)
    key = jax.random.PRNGKey(0)

    kv_cache = {
        "c_k": jnp.zeros((num_layers, 1, seq_len, CONFIG["latent_dim"]), dtype=jnp.float32),
        "c_v": jnp.zeros((num_layers, 1, seq_len, CONFIG["latent_dim"]), dtype=jnp.float32),
        "k_rope": jnp.zeros((num_layers, 1, seq_len, rope_dim), dtype=jnp.float32),
    }
    
    cos_full, sin_full = precompute_rope_freqs(seq_len, head_dim)

    for step in range(len(cur_ids) - 1):
        _, kv_cache = generation_step_cached(
            params, kv_cache, jnp.array(step), jnp.array([[cur_ids[step]]]),
            seq_len, CONFIG["num_heads"], num_kv_heads, head_dim, CONFIG["eps_rms"],
            cos_full, sin_full
        )

    for _ in range(length):
        T = len(cur_ids)
        if T >= seq_len: break

        logits, kv_cache = generation_step_cached(
            params, kv_cache, jnp.array(T - 1), jnp.array([[cur_ids[-1]]]),
            seq_len, CONFIG["num_heads"], num_kv_heads, head_dim, CONFIG["eps_rms"],
            cos_full, sin_full
        )

        logits = logits.squeeze(0)
        if repetition_penalty is not None and T > 0:
            logits = logits.at[jnp.unique(jnp.array(cur_ids))].set(
                logits[jnp.unique(jnp.array(cur_ids))] / repetition_penalty
            )
        logits = logits / temperature

        if top_k is not None and top_k > 0:
            logits = jnp.where(logits < jnp.sort(logits)[-top_k], -1e10, logits)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_idx = jnp.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            cumsum = jnp.cumsum(jax.nn.softmax(sorted_logits))
            first_cut = jnp.argmax(jnp.where(cumsum > top_p, 1, 0))
            mask = jnp.ones_like(sorted_logits, dtype=bool).at[first_cut + 1:].set(False)
            logits = jnp.zeros_like(logits).at[sorted_idx].set(jnp.where(mask, sorted_logits, -1e10))

        key, sub = jax.random.split(key)
        next_token = int(jax.random.categorical(sub, logits))
        if T < 10: print(f"  Step {T}: token_id={next_token}, piece='{sp.id_to_piece(next_token)}'")
        cur_ids.append(next_token)
        if next_token == sp.eos_id(): break

    return sp.decode(cur_ids[len(input_ids):])

# -------------------------------------------------------------------
# Initialization (correct weight shapes for decoupled RoPE)
# -------------------------------------------------------------------
def init_params(key, vocab_size, d_model, num_layers, num_heads, num_kv_heads):
    head_dim = d_model // num_heads
    rope_dim = head_dim // 2
    content_dim = head_dim - rope_dim
    
    params = {
        "tok_embeddings": {"weight": jax.random.normal(key, (vocab_size, d_model)) * 0.02},
        "transformer": {
            "layers": [],
            "rms_final": {"scale": jnp.zeros((d_model,), dtype=jnp.float32)},
        },
    }
    
    for _ in range(num_layers):
        key, k_q, k_o, k_gate, k_up, k_down, k_ck, k_cv, k_uk, k_uv, k_rope = jax.random.split(key, 11)
        layer = {
            "self_attn": {
                "q_proj": jax.random.normal(k_q, (d_model, num_heads * head_dim)) * 0.02,
                "o_proj": jax.random.normal(k_o, (d_model, d_model)) * 0.02,
            },
            "mlp": {
                "gate_proj": jax.random.normal(k_gate, (d_model, CONFIG["ffn_hidden"])) * 0.02,
                "up_proj":   jax.random.normal(k_up,   (d_model, CONFIG["ffn_hidden"])) * 0.02,
                "down_proj": jax.random.normal(k_down, (CONFIG["ffn_hidden"], d_model)) * 0.02,
            },
            "rms_attn": {"scale": jnp.zeros((d_model,), dtype=jnp.float32)},
            "rms_ffn":  {"scale": jnp.zeros((d_model,), dtype=jnp.float32)},
            "mla": {
                "w_ck":   jax.random.normal(k_ck,   (d_model, CONFIG["latent_dim"])) * 0.02,
                "w_cv":   jax.random.normal(k_cv,   (d_model, CONFIG["latent_dim"])) * 0.02,
                "w_uk":   jax.random.normal(k_uk,   (CONFIG["latent_dim"], num_kv_heads * content_dim)) * 0.02,
                "w_uv":   jax.random.normal(k_uv,   (CONFIG["latent_dim"], num_kv_heads * head_dim)) * 0.02,
                "w_rope": jax.random.normal(k_rope, (d_model, rope_dim)) * 0.02,
            }
        }
        params["transformer"]["layers"].append(layer)
    return params