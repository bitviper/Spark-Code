# spark_math_research_loop.py
"""
Closed‑loop research pipeline for character‑level arithmetic.
Train from scratch, resume training, or run inference only.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.75"
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import pickle, random, jax, jax.numpy as jnp, numpy as np
from tqdm import tqdm

import spark_llm_f as spark_llm
from spark_auditor_f import SparkAuditor

# --------------------------------------------------------------------
# 1. Character‑level tokenizer for arithmetic
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# 2. Tiny model configuration
# --------------------------------------------------------------------
spark_llm.CONFIG.update({
    "d_model": 128,
    "num_layers": 4,
    "num_heads": 4,
    "num_kv_heads": 2,
    "ffn_hidden": 512,
    "seq_len": 32,
    "vocab_size": tokenizer.vocab_size,
    "lr": 1e-3,               # FIX: Reverted to stable LR
    "batch_size": 32,         # FIX: Increased batch size for smoother gradients
    "chunk_size": 20,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "dropout_rate": 0.1,
})

# --------------------------------------------------------------------
# 3. Data generation (Zero-Padded for easier learning)
# --------------------------------------------------------------------
def generate_equations(num_range, num_samples, ops=['+', '-']):
    pairs = []
    for _ in range(num_samples):
        a = random.randint(*num_range)
        b = random.randint(*num_range)
        op = random.choice(ops)
        if op == '-':
            if b > a: a, b = b, a
            result = a - b
        else:
            result = a + b
        inp = f"{a:02d}{op}{b:02d}=?"
        out = f"{result:02d}?"
        pairs.append((inp, out))
    return pairs

def tokens_from_pairs(pairs):
    all_ids = []
    for inp, out in pairs:
        all_ids.extend(tokenizer.encode(inp + out))
    return jnp.array(all_ids, dtype=jnp.int32)

# --------------------------------------------------------------------
# 4. Prediction helper (greedy autoregressive)
# --------------------------------------------------------------------
def predict(params, equation_str):
    ids = tokenizer.encode(equation_str)
    cur_ids = list(ids)
    head_dim = spark_llm.CONFIG["d_model"] // spark_llm.CONFIG["num_heads"]
    num_layers = spark_llm.CONFIG["num_layers"]
    num_kv_heads = spark_llm.CONFIG["num_kv_heads"]
    seq_len = spark_llm.CONFIG["seq_len"]

    kv_cache = {
        "k": jnp.zeros((num_layers, 1, num_kv_heads, seq_len, head_dim), dtype=jnp.float32),
        "v": jnp.zeros((num_layers, 1, num_kv_heads, seq_len, head_dim), dtype=jnp.float32)
    }

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

# --------------------------------------------------------------------
# 5. Categorisation of equations for targeted repair
# --------------------------------------------------------------------
def categorize_equation(inp):
    if '+' in inp:
        parts = inp.split('+')
        if len(parts) == 2:
            try:
                a = int(parts[0])
                b = int(parts[1].rstrip('=?'))
                if (a % 10) + (b % 10) >= 10:
                    return "addition_carry"
                else:
                    return "addition_nocarry"
            except:
                return "addition"
        return "addition"
    elif '-' in inp:
        parts = inp.split('-')
        if len(parts) == 2:
            try:
                a = int(parts[0])
                b = int(parts[1].rstrip('=?'))
                if (a % 10) < (b % 10):
                    return "subtraction_borrow"
                else:
                    return "subtraction_noborrow"
            except:
                return "subtraction"
        return "subtraction"
    return "other"

# --------------------------------------------------------------------
# 6. Probe: per‑category accuracy + overall score
# --------------------------------------------------------------------
def probe_accuracy(params, val_pairs):
    correct = 0
    total = 0
    category_correct = {}
    category_total = {}

    for inp, expected in val_pairs[:100]: # Kept at 100 for speed
        pred = predict(params, inp)
        cat = categorize_equation(inp)
        category_total[cat] = category_total.get(cat, 0) + 1
        
        try:
            if int(pred) == int(expected.rstrip('?')):
                correct += 1
                category_correct[cat] = category_correct.get(cat, 0) + 1
        except:
            pass
        total += 1

    overall = (correct / total) * 100.0 if total > 0 else 0.0
    cat_acc = {cat: (category_correct.get(cat, 0) / category_total[cat]) * 100.0
               for cat in category_total}
    return overall, cat_acc

# --------------------------------------------------------------------
# 7. Fix appendix: targeted generators for each category
# --------------------------------------------------------------------
def generate_addition_carry(num=30):
    pairs = []
    for _ in range(num):
        a = random.randint(5, 15)
        b = random.randint(5, 15)
        result = a + b
        inp = f"{a:02d}+{b:02d}=?"
        out = f"{result:02d}?"
        pairs.append((inp, out))
    return pairs

def generate_subtraction_borrow(num=30):
    pairs = []
    for _ in range(num):
        while True:
            a = random.randint(10, 20)
            if a % 10 <= 8:      # ensures a borrow can happen
                break
        b = random.randint(1, 9)
        while (a % 10) >= b:
            b = random.randint(1, 9)
        result = a - b
        inp = f"{a:02d}-{b:02d}=?"
        out = f"{result:02d}?"
        pairs.append((inp, out))
    return pairs

FIX_FUNCTIONS = {
    "addition_carry": generate_addition_carry,
    "subtraction_borrow": generate_subtraction_borrow,
    "addition_nocarry": lambda n=30: generate_equations((5, 20), n, ops=['+']),
    "subtraction_noborrow": lambda n=30: generate_equations((5, 20), n, ops=['-']),
    "addition": lambda n=30: generate_equations((5, 20), n, ops=['+']),
    "subtraction": lambda n=30: generate_equations((5, 20), n, ops=['-']),
    "other": lambda n=30: generate_equations((5, 20), n, ops=['+', '-']),
}

# --------------------------------------------------------------------
# 8. Training loop (encapsulated)
# --------------------------------------------------------------------
def run_training(params, config, train_data, val_ids, val_pairs,
                 start_chunk=0, steps=2000):
    PROBE_INTERVAL    = 50
    REPAIR_THRESHOLD  = 80.0
    PROBE_WARMUP      = 30
    initial_train_size = len(train_data)

    auditor = SparkAuditor(max_grad_norm=config["max_grad_norm"],
                           lr_throttle_threshold=5.0,
                           anomaly_threshold=1.5)
    auditor.best_val_loss = float('inf')
    auditor.val_window_size = 3
    auditor.overfit_delta = 0.05

    key = jax.random.PRNGKey(123)
    opt_state = None
    opt_path = CHECKPOINT + ".opt"
    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            opt_state = pickle.load(f)
    else:
        opt_state = spark_llm.init_optimizer_state(params)

    cat_history = {}
    best_params = None

    def compute_val_loss(p, step):
        B, T = 4, config["seq_len"]
        idx = jax.random.randint(jax.random.PRNGKey(step), (B,), 0, len(val_ids)-T-1)
        batch = jax.vmap(lambda s: jax.lax.dynamic_slice(val_ids, (s,), (T,)))(idx)
        x, y = batch[:, :-1], batch[:, 1:]
        logits = spark_llm.forward(p, x, T, config["num_heads"],
                                   config["num_kv_heads"],
                                   config["d_model"]//config["num_heads"],
                                   config["eps_rms"], False)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    total_chunks = steps // config["chunk_size"]
    print(f"Starting training: {total_chunks} chunks (from chunk {start_chunk+1})")
    for chunk in tqdm(range(start_chunk, total_chunks), desc="Arithmetic loop"):
        (params, opt_state, key), (losses, grad_norms) = spark_llm.train_chunk(
            params, opt_state, key, config["chunk_size"], train_data
        )
        loss = float(losses[-1])
        grad = float(grad_norms[-1])
        
        if (chunk + 1) % 20 == 0:
            val_loss = float(compute_val_loss(params, chunk + 1))
            auditor.val_loss_history.append(val_loss)
            tqdm.write(f"Chunk {chunk+1}: loss={loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < auditor.best_val_loss:
                auditor.best_val_loss = val_loss
                best_params = jax.tree.map(lambda p: p, params)
                
            if len(auditor.val_loss_history) >= auditor.val_window_size + 1:
                recent = auditor.val_loss_history[-auditor.val_window_size:]
                if all(v > auditor.best_val_loss + auditor.overfit_delta for v in recent):
                    print(f"\n🛑 Overfitting detected at chunk {chunk+1}. Stopping early.")
                    break
            sys.stdout.flush()

        if (chunk + 1) >= PROBE_WARMUP and (chunk + 1) % PROBE_INTERVAL == 0:
            overall_acc, cat_acc = probe_accuracy(params, val_pairs)
            auditor.update_probe_scores({"overall": overall_acc, **cat_acc})

            print(f"📈 Probe at chunk {chunk+1}: overall={overall_acc:.1f}%")
            for cat, acc in sorted(cat_acc.items()):
                print(f"   {cat}: {acc:.1f}%")

            if overall_acc < 2.0:
                continue

            PATIENCE = 2  
            for cat, acc in cat_acc.items():
                if acc < REPAIR_THRESHOLD:
                    hist = cat_history.setdefault(cat, [])
                    hist.append((chunk + 1, acc))
                    if len(hist) > PATIENCE + 1:
                        hist.pop(0)

                    if len(hist) >= 2 and hist[-1][1] <= hist[0][1]:
                        print(f"🔧 Repairing '{cat}' (stalled at {acc:.1f}%)…")
                        if cat in FIX_FUNCTIONS:
                            new_pairs = FIX_FUNCTIONS[cat](40)
                            new_ids = tokens_from_pairs(new_pairs)
                            train_data = jnp.concatenate([train_data, new_ids])[:initial_train_size]
                elif acc >= REPAIR_THRESHOLD:
                    cat_history.pop(cat, None)
            sys.stdout.flush()

        if (chunk + 1) % 100 == 0:
            cpu_params = jax.device_get(params)
            cpu_opt_state = jax.device_get(opt_state)
            with open(CHECKPOINT, "wb") as f:
                pickle.dump({"params": cpu_params, "config": config, "chunk": chunk+1}, f)
            with open(opt_path, "wb") as f:
                pickle.dump(cpu_opt_state, f)

    if best_params is not None:
        print("✅ Restored best model parameters based on validation loss.")
        params = best_params

    cpu_params = jax.device_get(params)
    cpu_opt_state = jax.device_get(opt_state)
    with open(CHECKPOINT, "wb") as f:
        pickle.dump({"params": cpu_params, "config": config, "chunk": total_chunks}, f)
    with open(opt_path, "wb") as f:
        pickle.dump(cpu_opt_state, f)
    print("✅ Training complete.")
    return params

# --------------------------------------------------------------------
# 9. Interactive inference (with multi‑layer heatmaps)
# --------------------------------------------------------------------
def show_attention(params, equation_str, hide_eos=True):
    ids = tokenizer.encode(equation_str)
    T = min(len(ids), spark_llm.CONFIG["seq_len"])
    padded = ids[:T] + [0] * (spark_llm.CONFIG["seq_len"] - T)
    x = jnp.array([padded], dtype=jnp.int32)

    logits, attn_maps = spark_llm.forward(
        params, x,
        spark_llm.CONFIG["seq_len"],
        spark_llm.CONFIG["num_heads"],
        spark_llm.CONFIG["num_kv_heads"],
        spark_llm.CONFIG["d_model"] // spark_llm.CONFIG["num_heads"],
        spark_llm.CONFIG["eps_rms"],
        return_internals=True
    )

    all_labels = [tokenizer.id_to_char.get(t, '?') for t in padded[:T]]

    if hide_eos:
        keep_idx = [i for i, lab in enumerate(all_labels) if lab != '?']
        if not keep_idx:
            print("(No tokens left to display)")
            return
        labels = [all_labels[i] for i in keep_idx]
        eq_str = "".join(labels)
    else:
        keep_idx = list(range(T))
        labels = all_labels
        eq_str = "".join(all_labels)

    K = len(keep_idx)
    num_layers = len(attn_maps)

    def render_block(attn_map):
        full = np.array(jnp.mean(attn_map[0], axis=0))
        sub = full[np.ix_(keep_idx, keep_idx)]
        rows = []
        for i in range(K):
            row = f"{labels[i]:>3} | "
            for j in range(K):
                w = sub[i, j]
                if w < 0.05:     row += "."
                elif w < 0.20:   row += "*"
                elif w < 0.50:   row += "#"
                else:            row += "█"
            rows.append(row)
        return rows

    blocks = [render_block(m) for m in attn_maps]

    ROW_LABEL_WIDTH = 6
    SPACER = "   "

    top_line = " " * ROW_LABEL_WIDTH
    for idx in range(num_layers):
        top_line += eq_str.ljust(K)
        if idx < num_layers - 1:
            top_line += SPACER
    print(top_line)

    for r in range(K):
        line = ""
        for idx in range(num_layers):
            line += blocks[idx][r]
            if idx < num_layers - 1:
                line += SPACER
        print(line)

    bottom_line = " " * ROW_LABEL_WIDTH
    for idx in range(num_layers):
        bottom_line += f"Layer {idx+1}".center(K)
        if idx < num_layers - 1:
            bottom_line += SPACER
    print(bottom_line)
    print("-" * (len(top_line) + 2))

def run_inference(params):
    print("\n" + "="*60)
    print("INTERACTIVE ARITHMETIC TEST")
    print("Type an equation like '7+8' or 'exit' to quit.")
    print("Type 'heatmap on' / 'heatmap off' to toggle auto‑heatmaps.\n")

    auto_heatmap = False
    while True:
        try:
            user_in = input("> ").strip()
        except EOFError:
            break
            
        if user_in.lower() == 'exit':
            break
        if user_in.lower() == 'heatmap on':
            auto_heatmap = True
            print("Auto‑heatmap: ON")
            continue
        if user_in.lower() == 'heatmap off':
            auto_heatmap = False
            print("Auto‑heatmap: OFF")
            continue
        if not user_in:
            continue

        if user_in.endswith("=?"):
            core = user_in[:-2]
        elif user_in.endswith("="):
            core = user_in[:-1]
        elif user_in.endswith("?"):
            core = user_in[:-1]
        else:
            core = user_in

        op = '+' if '+' in core else ('-' if '-' in core else None)
        if op is None:
            print("  Invalid format. Use e.g. 7+8 or 12-5")
            continue

        parts = core.split(op)
        if len(parts) != 2:
            print("  Invalid format.")
            continue

        try:
            a = int(parts[0])
            b = int(parts[1])
        except:
            print("  Invalid numbers.")
            continue

        padded_input = f"{a:02d}{op}{b:02d}=?"
        padded_answer = predict(params, padded_input)
        
        try:
            clean_answer = str(int(padded_answer))
        except:
            clean_answer = padded_answer

        print(f"  {a}{op}{b} = {clean_answer}")
        
        if auto_heatmap:
            show_attention(params, padded_input + padded_answer)
        else:
            try:
                show = input("Show attention heatmap? (y/n): ").strip().lower().startswith('y')
            except EOFError:
                break
            if show:
                show_attention(params, padded_input + padded_answer)

# --------------------------------------------------------------------
# 10. Main – choose mode
# --------------------------------------------------------------------
CHECKPOINT = "spark_math_checkpoint.pkl"

def main():
    print("=" * 50)
    print("  SPARK ARITHMETIC RESEARCH PIPELINE")
    print("=" * 50)
    print("Choose mode:")
    print("  a - Train from scratch")
    print("  b - Resume training")
    print("  c - Inference only (load checkpoint)")
    
    try:
        choice = input("> ").strip().lower()
    except EOFError:
        return

    if choice == 'c':
        if not os.path.exists(CHECKPOINT):
            print("❌ No checkpoint found. Run training first.")
            return
        with open(CHECKPOINT, "rb") as f:
            ckpt = pickle.load(f)
        params = ckpt["params"]
        config = ckpt.get("config", spark_llm.CONFIG)
        spark_llm.CONFIG.update(config)
        run_inference(params)
        return

    if choice == 'a':
        print("Initialising fresh model...")
        params = spark_llm.init_params(
            jax.random.PRNGKey(0),
            tokenizer.vocab_size,
            spark_llm.CONFIG["d_model"],
            spark_llm.CONFIG["num_layers"],
            spark_llm.CONFIG["num_heads"],
            spark_llm.CONFIG["num_kv_heads"]
        )
        start_chunk = 0
    elif choice == 'b':
        if not os.path.exists(CHECKPOINT):
            print("❌ No checkpoint found. Starting fresh.")
            params = spark_llm.init_params(
                jax.random.PRNGKey(0),
                tokenizer.vocab_size,
                spark_llm.CONFIG["d_model"],
                spark_llm.CONFIG["num_layers"],
                spark_llm.CONFIG["num_heads"],
                spark_llm.CONFIG["num_kv_heads"]
            )
            start_chunk = 0
        else:
            with open(CHECKPOINT, "rb") as f:
                ckpt = pickle.load(f)
            params = ckpt["params"]
            config = ckpt.get("config", {})
            spark_llm.CONFIG.update(config)
            start_chunk = ckpt.get("chunk", 0)
            print(f"Resumed from chunk {start_chunk}")
    else:
        print("Invalid choice.")
        return

    print("Generating arithmetic data...")
    train_pairs = generate_equations((1, 20), 20000)
    val_pairs   = generate_equations((1, 20), 400)
    train_ids = tokens_from_pairs(train_pairs)
    val_ids   = tokens_from_pairs(val_pairs)
    train_data = jax.device_put(train_ids)

    params = run_training(params, spark_llm.CONFIG, train_data, val_ids, val_pairs,
                          start_chunk=start_chunk, steps=6000)

    try:
        ans = input("Start interactive test? (y/n): ").strip().lower()
        if ans.startswith('y'):
            run_inference(params)
    except EOFError:
        pass

if __name__ == "__main__":
    main()