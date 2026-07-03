# test_math_model.py
"""Interactive test for the trained arithmetic model."""

import os
import pickle, jax, jax.numpy as jnp, numpy as np

# Updated to use the refactored framework modules
import spark_llm_f as spark_llm

# -------------------------------
# 1. Recreate the tokenizer (must match training)
# -------------------------------
class MathCharTokenizer:
    def __init__(self):
        chars = "0123456789+-= ?"
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.eos_id = self.char_to_id['?']
        self.vocab_size = len(self.char_to_id)
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)

tokenizer = MathCharTokenizer()

# -------------------------------
# 2. Load the checkpoint (the latest one you saved)
# -------------------------------
# FIX: Aligning checkpoint name with the training script
CHECKPOINT = "spark_math_checkpoint.pkl"   

if not os.path.exists(CHECKPOINT):
    print(f"❌ Error: Checkpoint '{CHECKPOINT}' not found.")
    print("Please train the model using spark_math_research_loop.py first.")
    exit()

with open(CHECKPOINT, "rb") as f:
    ckpt = pickle.load(f)
params = ckpt["params"]
# Restore the exact config used during training
spark_llm.CONFIG.update(ckpt.get("config", {}))

# -------------------------------
# 3. Prediction function (same as in training)
# -------------------------------
def predict(equation_str):
    ids = tokenizer.encode(equation_str)
    cur_ids = list(ids)
    head_dim = spark_llm.CONFIG["d_model"] // spark_llm.CONFIG["num_heads"]
    num_layers = spark_llm.CONFIG["num_layers"]
    num_kv_heads = spark_llm.CONFIG["num_kv_heads"]
    seq_len = spark_llm.CONFIG["seq_len"]

    # FIX: Added dtype=jnp.float32 to prevent tracing mismatch
    kv_cache = {
        "k": jnp.zeros((num_layers, 1, num_kv_heads, seq_len, head_dim), dtype=jnp.float32),
        "v": jnp.zeros((num_layers, 1, num_kv_heads, seq_len, head_dim), dtype=jnp.float32)
    }

    # prime cache with input tokens (all except final '?')
    for step, tok in enumerate(ids[:-1]):
        single = jnp.array([[tok]], dtype=jnp.int32)
        _, kv_cache = spark_llm.generation_step_cached(
            params, kv_cache, jnp.array(step, dtype=jnp.int32), single,
            seq_len, spark_llm.CONFIG["num_heads"], num_kv_heads, head_dim,
            spark_llm.CONFIG["eps_rms"]
        )

    generated = []
    current_token = ids[-1]
    for step in range(len(ids)-1, seq_len):
        single = jnp.array([[current_token]], dtype=jnp.int32)
        logits, kv_cache = spark_llm.generation_step_cached(
            params, kv_cache, jnp.array(step, dtype=jnp.int32), single,
            seq_len, spark_llm.CONFIG["num_heads"], num_kv_heads, head_dim,
            spark_llm.CONFIG["eps_rms"]
        )
        next_token = int(jnp.argmax(logits[0]))
        generated.append(next_token)
        if next_token == tokenizer.eos_id:
            break
        current_token = next_token
    return tokenizer.decode(generated).rstrip('?')

# -------------------------------
# 4. Interactive loop
# -------------------------------
print("Arithmetic Model Interactive Test")
print("Type an equation like '7+8=?' or '12-5=?' (numbers 1-12 work best)")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("> ").strip()
    if user_input.lower() == 'exit':
        break
    if not user_input:
        continue
    # Ensure it ends with =?
    if not user_input.endswith("=?"):
        user_input = user_input.rstrip("?") + "=?"
    answer = predict(user_input)
    # Clean formatting to match the prompt cleanly
    clean_input = user_input.rstrip('?')
    print(f"  {clean_input}={answer}?")