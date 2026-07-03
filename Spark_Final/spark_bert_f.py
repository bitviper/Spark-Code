# spark_bert_f.py
# Bidirectional (BERT‑style) transformer with correct dropout threading.
# Uses learned absolute positional embeddings instead of RoPE.
#
# Critical fixes:
# - No RoPE (ensures true bidirectionality)
# - Dropout key properly threaded through forward pass
# - Independent PRNG keys for batch selection, MLM masking, and dropout
# - Mask token ID enforced via config

import jax
import jax.numpy as jnp
from functools import partial
import pickle
import os

from spark_llm_f import (
    CONFIG as LLM_CONFIG,            
    rms_norm,
    apply_dropout,
    mlp,
    init_optimizer_state,
    apply_adamw_update,
    clip_grads,
    save_checkpoint,
    load_checkpoint,
)

CONFIG = LLM_CONFIG.copy()
CONFIG.update({
    "mask_token_id": 4999,  
    "dropout_rate": 0.1,    # Essential for BERT
})

# -------------------------------------------------------------------
# Bidirectional attention
# -------------------------------------------------------------------
def attention(x, layer, num_heads, num_kv_heads, head_dim,
              mask=None, return_weights=False, key=None, dropout_rate=0.0):
    B, T, D = x.shape
    qkv = x @ layer["self_attn"]["qkv_proj"]
    q_end = num_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim

    q = qkv[:, :, :q_end].reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = qkv[:, :, q_end:k_end].reshape(B, T, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = qkv[:, :, k_end:].reshape(B, T, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    k_rep = jnp.repeat(k, num_heads // num_kv_heads, axis=1)
    v_rep = jnp.repeat(v, num_heads // num_kv_heads, axis=1)
    logits = (q @ k_rep.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)

    if mask is not None:
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

    attn_weights = jax.nn.softmax(logits, axis=-1)

    if key is not None and dropout_rate > 0.0:
        attn_weights = apply_dropout(attn_weights, dropout_rate, key)

    out = attn_weights @ v_rep
    attn_out = out.transpose(0, 2, 1, 3).reshape(B, T, D) @ layer["self_attn"]["o_proj"]
    return attn_out, attn_weights if return_weights else None

# -------------------------------------------------------------------
# BERT forward pass (learned positional embeddings, full mask)
# -------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2,3,4,5,6,8))
def forward(params, input_ids, seq_len, num_heads, num_kv_heads, head_dim, 
            eps_rms, dropout_key, return_internals=False):
    B, T = input_ids.shape
    tok_emb = params["tok_embeddings"]["weight"]
    pos_emb = params["pos_embeddings"]["weight"][:T, :]
    
    x = tok_emb[input_ids] + pos_emb

    num_layers = len(params["transformer"]["layers"])
    if CONFIG["dropout_rate"] > 0.0 and dropout_key is not None:
        keys = jax.random.split(dropout_key, 1 + num_layers * 2)
        x = apply_dropout(x, CONFIG["dropout_rate"], keys[0])
        layer_keys = keys[1:]
    else:
        layer_keys = [None] * (num_layers * 2)

    mask = jnp.ones((1, 1, T, T), dtype=bool)
    attention_maps = [] if return_internals else None

    for i, layer in enumerate(params["transformer"]["layers"]):
        attn_out, weights = attention(
            rms_norm(x, layer["rms_attn"]["scale"], eps_rms),
            layer, num_heads, num_kv_heads, head_dim, mask,
            return_weights=return_internals,
            key=layer_keys[i*2],
            dropout_rate=CONFIG["dropout_rate"]
        )
        x = x + attn_out
        if return_internals:
            attention_maps.append(weights)

        x = x + mlp(
            rms_norm(x, layer["rms_ffn"]["scale"], eps_rms),
            layer,
            key=layer_keys[i*2 + 1],
            dropout_rate=CONFIG["dropout_rate"]
        )

    x = rms_norm(x, params["transformer"]["rms_final"]["scale"], eps_rms)
    logits = x @ tok_emb.T

    if return_internals:
        return logits, attention_maps
    return logits

# -------------------------------------------------------------------
# MLM Batch Creation
# -------------------------------------------------------------------
def create_mlm_batch(ids, mask_prob=0.25, rng_key=None, vocab_size=None, mask_token_id=None):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    if vocab_size is None:
        vocab_size = CONFIG["vocab_size"]
    if mask_token_id is None:
        mask_token_id = CONFIG["mask_token_id"]

    B, T = ids.shape
    rng_key, sub = jax.random.split(rng_key)
    mask_sel = jax.random.uniform(sub, (B, T)) < mask_prob

    input_ids = ids.copy()
    rng_key, sub1, sub2 = jax.random.split(rng_key, 3)
    mask_type = jax.random.uniform(sub1, (B, T))

    mask_mask = mask_sel & (mask_type < 0.8)
    input_ids = jnp.where(mask_mask, mask_token_id, input_ids)

    rand_mask = mask_sel & (mask_type >= 0.8) & (mask_type < 0.9)
    rand_tokens = jax.random.randint(sub2, (B, T), 0, vocab_size)
    input_ids = jnp.where(rand_mask, rand_tokens, input_ids)

    target_ids = jnp.where(mask_sel, ids, 0)
    return input_ids, target_ids, mask_sel

# -------------------------------------------------------------------
# MLM Loss
# -------------------------------------------------------------------
def mlm_loss(params, input_ids, target_ids, mlm_mask, config, dropout_key):
    logits = forward(
        params, input_ids,
        config["seq_len"], config["num_heads"], config["num_kv_heads"],
        config["d_model"] // config["num_heads"], config["eps_rms"], 
        dropout_key, False
    )
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    vocab_dim = logits.shape[-1]
    one_hot = jax.nn.one_hot(target_ids, vocab_dim)
    token_logp = jnp.sum(one_hot * log_probs, axis=-1)
    masked_logp = token_logp * mlm_mask
    loss = -jnp.sum(masked_logp) / jnp.maximum(jnp.sum(mlm_mask), 1.0)
    return loss

# -------------------------------------------------------------------
# Training step (with isolated PRNG keys)
# -------------------------------------------------------------------
def train_step_mlm(params, opt_state, key, ids, config):
    B, T = config["batch_size"], config["seq_len"]
    key, k_idx, k_mlm, k_drop = jax.random.split(key, 4)

    idx = jax.random.randint(k_idx, (B,), 0, ids.shape[0] - T - 1)
    batch = jax.vmap(lambda start: jax.lax.dynamic_slice(ids, (start,), (T,)))(idx)

    actual_vocab = params["tok_embeddings"]["weight"].shape[0]
    input_ids, target_ids, mlm_mask = create_mlm_batch(
        batch, mask_prob=0.25, rng_key=k_mlm, vocab_size=actual_vocab,
        mask_token_id=config["mask_token_id"]
    )

    def loss_fn(p, drop_key):
        return mlm_loss(p, input_ids, target_ids, mlm_mask, config, drop_key)

    loss, grads = jax.value_and_grad(loss_fn)(params, k_drop)
    grads, grad_norm = clip_grads(grads, max_norm=config["max_grad_norm"])
    new_params, new_opt_state = apply_adamw_update(
        params, grads, opt_state, config["lr"], config["adam_b1"],
        config["adam_b2"], config["adam_eps"], config["weight_decay"]
    )
    return new_params, new_opt_state, loss, grad_norm, key

@partial(jax.jit, static_argnums=(3,))
def train_chunk_mlm(params, opt_state, key, chunk_size, ids, config):
    def body(carry, _):
        p, o, k = carry
        p, o, loss, grad_norm, k = train_step_mlm(p, o, k, ids, config)
        return (p, o, k), (loss, grad_norm)
    final_carry, metrics = jax.lax.scan(body, (params, opt_state, key), None, length=chunk_size)
    return final_carry, metrics

# -------------------------------------------------------------------
# BERT‑specific initialization
# -------------------------------------------------------------------
def init_params_bert(key, vocab_size, d_model, num_layers, num_heads, num_kv_heads, seq_len):
    head_dim = d_model // num_heads
    ffn_hidden = CONFIG["ffn_hidden"]
    
    key, tok_key, pos_key = jax.random.split(key, 3)
    params = {
        "tok_embeddings": {
            "weight": jax.random.normal(tok_key, (vocab_size, d_model)) * 0.02
        },
        "pos_embeddings": {
            "weight": jax.random.normal(pos_key, (seq_len, d_model)) * 0.02
        },
        "transformer": {
            "layers": [],
            "rms_final": {"scale": jnp.zeros((d_model,), dtype=jnp.float32)},
        },
    }
    for _ in range(num_layers):
        key, k_qkv, k_o, k_gate, k_up, k_down = jax.random.split(key, 6)
        qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
        layer = {
            "self_attn": {
                "qkv_proj": jax.random.normal(k_qkv, (d_model, qkv_dim)) * 0.02,
                "o_proj":   jax.random.normal(k_o,   (d_model, d_model)) * 0.02,
            },
            "mlp": {
                "gate_proj": jax.random.normal(k_gate, (d_model, ffn_hidden)) * 0.02,
                "up_proj":   jax.random.normal(k_up,   (d_model, ffn_hidden)) * 0.02,
                "down_proj": jax.random.normal(k_down, (ffn_hidden, d_model)) * 0.02,
            },
            "rms_attn": {"scale": jnp.zeros((d_model,), dtype=jnp.float32)},
            "rms_ffn":  {"scale": jnp.zeros((d_model,), dtype=jnp.float32)},
        }
        params["transformer"]["layers"].append(layer)
    return params