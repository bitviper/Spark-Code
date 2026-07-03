# exercise_deepseek_code.py
"""
Exercise 3: Local Code Completion with MLA (spark_deepseek_f)
Train a tiny MLA transformer on synthetic Python functions and
demonstrate real-time code completion on the laptop GPU.
"""

import os, sys, random, pickle
import jax, jax.numpy as jnp
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import spark_deepseek_f as deepseek
from spark_data_f import build_tokenizer  # helper for sentencepiece

# --------------------------------------------------------------------
# Synthetic code generation (same as earlier)
# --------------------------------------------------------------------
FUNCTION_TEMPLATES = [
    "def {name}(a, b):\n    return a + b\n",
    "def {name}(a, b):\n    return a - b\n",
    "def {name}(a, b):\n    return a * b\n",
    "def {name}(a, b):\n    if b == 0:\n        return None\n    return a / b\n",
    "def {name}(x, y):\n    if x > y:\n        return x\n    else:\n        return y\n",
    "def {name}(x, y, z):\n    if x >= y and x >= z:\n        return x\n    elif y >= z:\n        return y\n    else:\n        return z\n",
    "def {name}(n):\n    result = []\n    for i in range(n):\n        result.append(i * i)\n    return result\n",
    "def {name}(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total\n",
    "def {name}(s, c):\n    return s.count(c)\n",
    "def {name}(s):\n    return s[::-1]\n",
    "def {name}(n):\n    if n <= 1:\n        return n\n    else:\n        return {name}(n-1) + {name}(n-2)\n",
]
NAMES = [
    "add", "subtract", "multiply", "divide", "max_of_two", "max_of_three",
    "square_list", "sum_list", "count_char", "reverse_string", "fibonacci",
    "factorial", "is_even", "gcd", "lcm", "power", "average", "median",
    "sort_descending", "filter_positive", "map_abs", "zip_with_add",
    "flatten_list", "unique_elements"
]

def generate_synthetic_code(num_samples=5000):
    samples = []
    for _ in range(num_samples):
        name = random.choice(NAMES)
        template = random.choice(FUNCTION_TEMPLATES)
        samples.append(template.format(name=name))
    return samples

# --------------------------------------------------------------------
# Tokenizer training (on synthetic code)
# --------------------------------------------------------------------
def train_tokenizer(output_dir="code_tok", vocab_size=2000):
    os.makedirs(output_dir, exist_ok=True)
    code = generate_synthetic_code(5000)
    corpus_path = os.path.join(output_dir, "code_corpus.txt")
    with open(corpus_path, "w") as f:
        for sample in code:
            f.write(sample + "\n")
    sp = build_tokenizer(
        corpus_path,
        model_prefix=os.path.join(output_dir, "code_spm"),
        vocab_size=vocab_size,
        user_defined_symbols=["<PAD>", "<UNK>", "<BOS>", "<EOS>", "🔲"]
    )
    return sp, code

# --------------------------------------------------------------------
# MLA configuration (fits GTX 1650)
# --------------------------------------------------------------------
deepseek.CONFIG.update({
    "d_model": 384,
    "num_layers": 8,
    "num_heads": 6,
    "num_kv_heads": 2,
    "ffn_hidden": 1536,
    "seq_len": 128,
    "latent_dim": 16,
    "vocab_size": 2000,          # will be updated after tokenizer
    "lr": 3e-4,
    "batch_size": 4,
    "chunk_size": 10,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "dropout_rate": 0.1,
})

# --------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------
def prepare_data(sp, code_samples):
    all_ids = []
    for sample in code_samples:
        ids = sp.encode(sample)
        ids.append(sp.eos_id())
        all_ids.extend(ids)
    return jnp.array(all_ids, dtype=jnp.int32)

# --------------------------------------------------------------------
# Training (using deepseek.train_chunk_mla)
# --------------------------------------------------------------------
def train_mla(params, opt_state, key, train_ids, steps=2000):
    chunk_size = deepseek.CONFIG["chunk_size"]
    total_chunks = steps // chunk_size
    for chunk in tqdm(range(total_chunks), desc="MLA training"):
        (params, opt_state, key), (losses, _) = deepseek.train_chunk_mla(
            params, opt_state, key, chunk_size, train_ids
        )
        loss = float(losses[-1])
        if (chunk+1) % 10 == 0:
            print(f"Chunk {chunk+1}: loss={loss:.4f}")
    return params, opt_state, key

# --------------------------------------------------------------------
# Generation (MLA caching)
# --------------------------------------------------------------------
def generate_code_mla(params, sp, prompt, max_new_tokens=64, temp=0.8):
    seq_len = deepseek.CONFIG["seq_len"]
    head_dim = deepseek.CONFIG["d_model"] // deepseek.CONFIG["num_heads"]
    rope_dim = head_dim // 2
    num_layers = deepseek.CONFIG["num_layers"]
    num_kv_heads = deepseek.CONFIG["num_kv_heads"]

    input_ids = sp.encode(prompt)[:seq_len - max_new_tokens]
    cur_ids = list(input_ids)
    key = jax.random.PRNGKey(0)

    kv_cache = {
        "c_k": jnp.zeros((num_layers, 1, seq_len, deepseek.CONFIG["latent_dim"])),
        "c_v": jnp.zeros((num_layers, 1, seq_len, deepseek.CONFIG["latent_dim"])),
        "k_rope": jnp.zeros((num_layers, 1, seq_len, rope_dim)),
    }
    cos_full, sin_full = deepseek.precompute_rope_freqs(seq_len, head_dim)

    # Prefill cache with prompt
    for step in range(len(cur_ids) - 1):
        single = jnp.array([[cur_ids[step]]], dtype=jnp.int32)
        step_tensor = jnp.array(step, dtype=jnp.int32)
        _, kv_cache = deepseek.generation_step_cached(
            params, kv_cache, step_tensor, single,
            seq_len, deepseek.CONFIG["num_heads"], num_kv_heads, head_dim,
            deepseek.CONFIG["eps_rms"], cos_full, sin_full
        )

    generated = []
    current_token = cur_ids[-1]
    for step in range(len(cur_ids)-1, seq_len):
        single = jnp.array([[current_token]], dtype=jnp.int32)
        step_tensor = jnp.array(step, dtype=jnp.int32)
        logits, kv_cache = deepseek.generation_step_cached(
            params, kv_cache, step_tensor, single,
            seq_len, deepseek.CONFIG["num_heads"], num_kv_heads, head_dim,
            deepseek.CONFIG["eps_rms"], cos_full, sin_full
        )
        logits = logits[0] / temp
        key, sub = jax.random.split(key)
        next_token = int(jax.random.categorical(sub, logits))
        generated.append(next_token)
        if next_token == sp.eos_id():
            break
        current_token = next_token

    return prompt + sp.decode(generated)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    print("Training tokenizer on synthetic code...")
    sp, code_samples = train_tokenizer(vocab_size=2000)
    deepseek.CONFIG["vocab_size"] = sp.vocab_size()

    # Prepare data
    train_ids = prepare_data(sp, code_samples[:4500])
    val_ids   = prepare_data(sp, code_samples[4500:])
    print(f"Training tokens: {len(train_ids)}")

    # Initialise MLA model
    key = jax.random.PRNGKey(123)
    params = deepseek.init_params(
        key,
        deepseek.CONFIG["vocab_size"],
        deepseek.CONFIG["d_model"],
        deepseek.CONFIG["num_layers"],
        deepseek.CONFIG["num_heads"],
        deepseek.CONFIG["num_kv_heads"]
    )
    opt_state = deepseek.init_optimizer_state(params)

    # Train
    params, _, _ = train_mla(params, opt_state, key, train_ids, steps=2000)

    # Demo completions
    print("\n--- MLA Code Completion Demo ---")
    prompts = [
        "def add(a, b):\n    ",
        "def square_list(n):\n    result = []\n    for i in range(n):\n        ",
        "def max_of_three(x, y, z):\n    ",
    ]
    for p in prompts:
        print(f"\nPrompt:\n{p}")
        completion = generate_code_mla(params, sp, p, max_new_tokens=40)
        print(f"Completion:\n{completion}")