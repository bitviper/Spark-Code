# spark_llm_f.py
# Causal (autoregressive) language model + shared transformer components.
# Spark_BERT imports the shared utilities from here.

import os
import jax
import jax.numpy as jnp
from functools import partial
import pickle
from jax import remat

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
CONFIG = {
    "d_model": 512,
    "num_layers": 12,
    "num_heads": 8,
    "num_kv_heads": 2,
    "ffn_hidden": 2048,
    "seq_len": 512,
    "ntk_scale": 1.0,
    "vocab_size": 5000,
    "eps_rms": 1e-6,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "dropout_rate": 0.0,
    "lr": 5e-5,
    "batch_size": 4,
    "chunk_size": 20,
    "adam_b1": 0.9,
    "adam_b2": 0.999,
    "adam_eps": 1e-8,
}

# -------------------------------------------------------------------
# RoPE (shared)
# -------------------------------------------------------------------
def precompute_rope_freqs(seq_len, head_dim, base=10000.0, ntk_scale=1.0):
    half = head_dim // 2
    scaled_base = base * (ntk_scale ** (half / (half - 2))) if half > 2 else base * ntk_scale
    positions = jnp.arange(seq_len)
    dims = jnp.arange(half)
    inv_freq = 1.0 / (scaled_base ** (dims / half))
    angles = jnp.outer(positions, inv_freq)
    cos = jnp.repeat(jnp.cos(angles), 2, axis=-1)[None, None, :, :]
    sin = jnp.repeat(jnp.sin(angles), 2, axis=-1)[None, None, :, :]
    return cos, sin

def apply_rope(x, cos, sin):
    even = x[..., ::2]
    odd  = x[..., 1::2]
    rotated = jnp.empty_like(x)
    rotated = rotated.at[..., ::2].set(-odd)
    rotated = rotated.at[..., 1::2].set(even)
    return x * cos + rotated * sin

# -------------------------------------------------------------------
# RMSNorm, Swish, Dropout (shared)
# -------------------------------------------------------------------
def rms_norm(x, scale, eps):
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(var + eps) * (1.0 + scale)

def swish(x):
    return x * jax.nn.sigmoid(x)

def apply_dropout(x, rate, key=None):
    if rate <= 0.0 or key is None:
        return x
    keep_prob = 1.0 - rate
    mask = jax.random.bernoulli(key, keep_prob, x.shape)
    return (x * mask) / keep_prob

# -------------------------------------------------------------------
# Causal Attention
# -------------------------------------------------------------------
def attention(x, layer, num_heads, num_kv_heads, head_dim, cos, sin,
              mask=None, return_weights=False, key=None, dropout_rate=0.0):
    B, T, D = x.shape
    qkv = x @ layer["self_attn"]["qkv_proj"]
    q_end = num_heads * head_dim
    k_end = q_end + num_kv_heads * head_dim

    q = qkv[:, :, :q_end].reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = qkv[:, :, q_end:k_end].reshape(B, T, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = qkv[:, :, k_end:].reshape(B, T, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

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
# MLP (shared)
# -------------------------------------------------------------------
def mlp(x, layer, key=None, dropout_rate=0.0):
    h = swish(x @ layer["mlp"]["gate_proj"]) * (x @ layer["mlp"]["up_proj"])
    if key is not None and dropout_rate > 0.0:
        h = apply_dropout(h, dropout_rate, key)
    return h @ layer["mlp"]["down_proj"]

# -------------------------------------------------------------------
# Transformer Layer (gradient checkpointing via decorator)
# -------------------------------------------------------------------
@partial(jax.remat, policy=jax.checkpoint_policies.nothing_saveable)
def transformer_layer_fn(layer, x_emb, cos, sin, mask_attn, k1, k2, eps_rms):
    attn_out, _ = attention(
        rms_norm(x_emb, layer["rms_attn"]["scale"], eps_rms),
        layer, CONFIG["num_heads"], CONFIG["num_kv_heads"],
        CONFIG["d_model"] // CONFIG["num_heads"], cos, sin, mask_attn,
        return_weights=False, key=k1, dropout_rate=CONFIG["dropout_rate"]
    )
    x_emb = x_emb + attn_out
    x_emb = x_emb + mlp(
        rms_norm(x_emb, layer["rms_ffn"]["scale"], eps_rms),
        layer, key=k2, dropout_rate=CONFIG["dropout_rate"]
    )
    return x_emb

# -------------------------------------------------------------------
# Causal forward pass
# -------------------------------------------------------------------
@partial(jax.jit, static_argnums=(2,3,4,5,6,7))
def forward(params, input_ids, seq_len, num_heads, num_kv_heads, head_dim,
            eps_rms, return_internals=False):
    B, T = input_ids.shape
    tok_emb = params["tok_embeddings"]["weight"]
    x = tok_emb[input_ids] * jnp.sqrt(CONFIG["d_model"])

    cos_full, sin_full = precompute_rope_freqs(seq_len, head_dim,
                                                ntk_scale=CONFIG["ntk_scale"])
    cos = cos_full[:, :, :T, :]
    sin = sin_full[:, :, :T, :]

    mask = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))

    attention_maps = [] if return_internals else None

    if return_internals:
        for layer in params["transformer"]["layers"]:
            attn_out, weights = attention(
                rms_norm(x, layer["rms_attn"]["scale"], eps_rms),
                layer, num_heads, num_kv_heads, head_dim, cos, sin, mask,
                return_weights=True
            )
            x = x + attn_out
            attention_maps.append(weights)
            x = x + mlp(rms_norm(x, layer["rms_ffn"]["scale"], eps_rms), layer)
    else:
        for layer in params["transformer"]["layers"]:
            x = transformer_layer_fn(layer, x, cos, sin, mask, None, None, eps_rms)

    x = rms_norm(x, params["transformer"]["rms_final"]["scale"], eps_rms)
    logits = x @ tok_emb.T

    if return_internals:
        return logits, attention_maps
    return logits

# -------------------------------------------------------------------
# Optimiser & utilities (shared)
# -------------------------------------------------------------------
def init_optimizer_state(params):
    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "mu": jax.tree.map(lambda p: jnp.zeros_like(p), params),
        "nu": jax.tree.map(lambda p: jnp.zeros_like(p), params),
    }

def apply_adamw_update(params, grads, state, lr, b1, b2, eps, weight_decay):
    step = state["step"] + 1
    new_mu = jax.tree.map(lambda m, g: b1 * m + (1 - b1) * g, state["mu"], grads)
    new_nu = jax.tree.map(lambda n, g: b2 * n + (1 - b2) * (g * g), state["nu"], grads)
    mu_hat = jax.tree.map(lambda m: m / (1.0 - jnp.power(b1, step)), new_mu)
    nu_hat = jax.tree.map(lambda n: n / (1.0 - jnp.power(b2, step)), new_nu)
    new_params = jax.tree.map(
        lambda p, m, n: p - lr * (m / (jnp.sqrt(n) + eps) + weight_decay * p),
        params, mu_hat, nu_hat
    )
    return new_params, {"step": step, "mu": new_mu, "nu": new_nu}

def clip_grads(grads, max_norm=1.0):
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
    factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    return jax.tree.map(lambda g: g * factor, grads), grad_norm

# -------------------------------------------------------------------
# Causal training step & chunk
# -------------------------------------------------------------------
def train_step(params, opt_state, key, ids):
    B, T = CONFIG["batch_size"], CONFIG["seq_len"]
    key, sub = jax.random.split(key)
    idx = jax.random.randint(sub, (B,), 0, ids.shape[0] - T - 1)
    x = jax.vmap(lambda s: jax.lax.dynamic_slice(ids, (s,), (T,)))(idx)
    y = jax.vmap(lambda s: jax.lax.dynamic_slice(ids, (s+1,), (T,)))(idx)

    key, drop_key = jax.random.split(key)

    def loss_fn(p, d_key):
        tok_emb = p["tok_embeddings"]["weight"]
        x_emb = tok_emb[x] * jnp.sqrt(CONFIG["d_model"])
        cos, sin = precompute_rope_freqs(T, CONFIG["d_model"] // CONFIG["num_heads"])
        mask_attn = jnp.tril(jnp.ones((1, 1, T, T), dtype=bool))

        num_layers = len(p["transformer"]["layers"])
        layer_keys = jax.random.split(d_key, num_layers * 2)

        for i in range(num_layers):
            k1, k2 = layer_keys[i*2], layer_keys[i*2+1]
            x_emb = transformer_layer_fn(
                p["transformer"]["layers"][i], x_emb, cos, sin, mask_attn,
                k1, k2, CONFIG["eps_rms"]
            )

        logits = rms_norm(x_emb, p["transformer"]["rms_final"]["scale"],
                        CONFIG["eps_rms"]) @ tok_emb.T
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target = jax.nn.one_hot(y, logits.shape[-1])
        loss = -jnp.mean(jnp.sum(target * log_probs, axis=-1))
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params, drop_key)
    grads, grad_norm = clip_grads(grads, CONFIG["max_grad_norm"])
    new_params, new_opt_state = apply_adamw_update(
        params, grads, opt_state, CONFIG["lr"], CONFIG["adam_b1"],
        CONFIG["adam_b2"], CONFIG["adam_eps"], CONFIG["weight_decay"]
    )
    return new_params, new_opt_state, loss, grad_norm, key

@partial(jax.jit, static_argnums=(3,))
def train_chunk(params, opt_state, key, chunk_size, ids):
    def body(carry, _):
        p, o, k = carry
        p, o, loss, grad_norm, k = train_step(p, o, k, ids)
        return (p, o, k), (loss, grad_norm)
    final_carry, metrics = jax.lax.scan(body, (params, opt_state, key), None, length=chunk_size)
    return final_carry, metrics

# -------------------------------------------------------------------
# Autoregressive generation (KV‑cache)
# -------------------------------------------------------------------
@partial(jax.jit, static_argnums=(4,5,6,7))
def generation_step_cached(params, kv_cache, step_idx, input_ids,
                           seq_len, num_heads, num_kv_heads, head_dim, eps_rms):
    B, T = input_ids.shape
    tok_emb = params["tok_embeddings"]["weight"]
    x = tok_emb[input_ids] * jnp.sqrt(CONFIG["d_model"])

    cos_full, sin_full = precompute_rope_freqs(seq_len, head_dim,
                                                ntk_scale=CONFIG["ntk_scale"])
    cos = jax.lax.dynamic_slice(cos_full, (0, 0, step_idx, 0), (1, 1, 1, head_dim))
    sin = jax.lax.dynamic_slice(sin_full, (0, 0, step_idx, 0), (1, 1, 1, head_dim))

    new_k_cache, new_v_cache = [], []

    for i, layer in enumerate(params["transformer"]["layers"]):
        norm_x = rms_norm(x, layer["rms_attn"]["scale"], eps_rms)
        qkv = norm_x @ layer["self_attn"]["qkv_proj"]

        q_end = num_heads * head_dim
        k_end = q_end + num_kv_heads * head_dim

        q = qkv[:, :, :q_end].reshape(B, 1, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = qkv[:, :, q_end:k_end].reshape(B, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = qkv[:, :, k_end:].reshape(B, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        layer_k = jax.lax.dynamic_update_slice(kv_cache["k"][i], k, (0, 0, step_idx, 0))
        layer_v = jax.lax.dynamic_update_slice(kv_cache["v"][i], v, (0, 0, step_idx, 0))
        new_k_cache.append(layer_k)
        new_v_cache.append(layer_v)

        k_rep = jnp.repeat(layer_k, num_heads // num_kv_heads, axis=1)
        v_rep = jnp.repeat(layer_v, num_heads // num_kv_heads, axis=1)

        logits = (q @ k_rep.transpose(0, 1, 3, 2)) / jnp.sqrt(head_dim)
        
        mask = (jnp.arange(seq_len) <= step_idx)[None, None, None, :]
        
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)

        out = jax.nn.softmax(logits, axis=-1) @ v_rep
        out = out.transpose(0, 2, 1, 3).reshape(B, 1, CONFIG["d_model"])

        x = x + out @ layer["self_attn"]["o_proj"]
        x = x + mlp(rms_norm(x, layer["rms_ffn"]["scale"], eps_rms), layer)

    x = rms_norm(x, params["transformer"]["rms_final"]["scale"], eps_rms)
    logits = x @ tok_emb.T

    next_cache = {"k": jnp.stack(new_k_cache), "v": jnp.stack(new_v_cache)}
    return logits[:, -1, :], next_cache

def generate_text(params, sp, prompt_text, length=80, temperature=0.95,
                  top_k=40, top_p=0.85, repetition_penalty=1.2):
    seq_len = CONFIG["seq_len"]
    head_dim = CONFIG["d_model"] // CONFIG["num_heads"]
    num_layers = CONFIG["num_layers"]
    num_kv_heads = CONFIG["num_kv_heads"]

    input_ids = sp.encode(prompt_text)
    input_ids = input_ids[: seq_len - 1]
    cur_ids = list(input_ids)

    key = jax.random.PRNGKey(0)

    kv_cache = {
        "k": jnp.zeros((num_layers, 1, num_kv_heads, seq_len, head_dim), dtype=jnp.float32),
        "v": jnp.zeros((num_layers, 1, num_kv_heads, seq_len, head_dim), dtype=jnp.float32),
    }

    for step in range(len(cur_ids) - 1):
        single_id = jnp.array([[cur_ids[step]]], dtype=jnp.int32)
        step_tensor = jnp.array(step, dtype=jnp.int32)
        _, kv_cache = generation_step_cached(
            params, kv_cache, step_tensor, single_id,
            seq_len, CONFIG["num_heads"], num_kv_heads, head_dim, CONFIG["eps_rms"]
        )

    for _ in range(length):
        T = len(cur_ids)
        if T >= seq_len:
            break

        latest_token_id = jnp.array([[cur_ids[-1]]], dtype=jnp.int32)
        current_step_idx = jnp.array(T - 1, dtype=jnp.int32)

        logits, kv_cache = generation_step_cached(
            params, kv_cache, current_step_idx, latest_token_id,
            seq_len, CONFIG["num_heads"], num_kv_heads, head_dim, CONFIG["eps_rms"]
        )

        logits = logits.squeeze(0)

        if repetition_penalty is not None and T > 0:
            used = jnp.array(cur_ids, dtype=jnp.int32)
            unique_ids = jnp.unique(used)
            logits = logits.at[unique_ids].set(logits[unique_ids] / repetition_penalty)

        logits = logits / temperature

        if top_k is not None and top_k > 0:
            kth = jnp.sort(logits)[-top_k]
            logits = jnp.where(logits < kth, -1e10, logits)

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_idx = jnp.argsort(logits)[::-1]
            sorted_logits = logits[sorted_idx]
            sorted_probs = jax.nn.softmax(sorted_logits)
            cumsum = jnp.cumsum(sorted_probs)
            cutoff = jnp.where(cumsum > top_p, 1, 0)
            first_cut = jnp.argmax(cutoff)
            mask = jnp.ones_like(sorted_logits, dtype=bool).at[first_cut + 1:].set(False)
            masked_logits = jnp.where(mask, sorted_logits, -1e10)
            logits = jnp.zeros_like(logits).at[sorted_idx].set(masked_logits)

        probs = jax.nn.softmax(logits)
        key, sub = jax.random.split(key)
        next_token = int(jax.random.categorical(sub, logits))
        cur_ids.append(next_token)

        if next_token == sp.eos_id():
            break

    gen_ids = cur_ids[len(input_ids):]
    return sp.decode(gen_ids)

# -------------------------------------------------------------------
# Parameter initialisation (shared) - RESTORED
# -------------------------------------------------------------------
def init_params(key, vocab_size, d_model, num_layers, num_heads, num_kv_heads):
    head_dim = d_model // num_heads
    ffn_hidden = CONFIG["ffn_hidden"]
    key, tok_key = jax.random.split(key)
    params = {
        "tok_embeddings": {
            "weight": jax.random.normal(tok_key, (vocab_size, d_model)) * 0.02
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

# -------------------------------------------------------------------
# Save / Load (shared)
# -------------------------------------------------------------------
def save_checkpoint(params, opt_state, fname="spark_checkpoint.pkl", config=None, step=0):
    with open(fname, "wb") as f:
        pickle.dump({
            "params": params, 
            "opt_state": opt_state,
            "config": config if config is not None else CONFIG,
            "step": step
        }, f)

def load_checkpoint(fname="spark_checkpoint.pkl"):
    if not os.path.exists(fname):
        return None, None
    with open(fname, "rb") as f:
        d = pickle.load(f)
        return d["params"], d.get("opt_state", None)