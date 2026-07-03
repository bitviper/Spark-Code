# exercise_bert_math_advanced.py
"""
Advanced Masked Arithmetic Solver (Bidirectional BERT)
- Handles +, -, and * 
- Handles variable length results (up to 4 digits: 99*99=9801)
- Upgraded model capacity (d_model=320, 8 layers)
- Memory Optimized (Gradient Accumulation for 4GB GPUs)
"""

import os, sys, random, math, pickle
import jax, jax.numpy as jnp
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import spark_bert_f as bert
from spark_auditor_f import SparkAuditor

# --------------------------------------------------------------------
# Character tokenizer (Added *)
# --------------------------------------------------------------------
class MathCharTokenizer:
    def __init__(self):
        chars = "0123456789+-*= ?"  # Added *
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.mask_id = self.char_to_id['?']
        self.eq_id = self.char_to_id['=']
        self.vocab_size = len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(int(i), '?') for i in ids)

tokenizer = MathCharTokenizer()

# --------------------------------------------------------------------
# Upgraded Configuration (Memory Optimized for ~4GB VRAM)
# --------------------------------------------------------------------
bert.CONFIG.update({
    "d_model": 320,          # Increased from 256, but lower than 384 to save VRAM
    "num_layers": 8,         # Increased from 6 (depth helps multiplication logic)
    "num_heads": 8,          # 320 / 8 = 40 dim per head
    "num_kv_heads": 8,
    "ffn_hidden": 1280,      # 4x d_model
    "seq_len": 32,
    "vocab_size": tokenizer.vocab_size,
    "mask_token_id": tokenizer.mask_id,
    "lr": 3e-4,              
    "batch_size": 64,        # Reduced from 256 to fit in 4GB VRAM!
    "chunk_size": 20,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "dropout_rate": 0.1,
    "warmup_steps": 1000,    
})
GRAD_ACCUM_STEPS = 4  # Simulates batch_size=256 (64 * 4) without the VRAM cost

# --------------------------------------------------------------------
# Checkpoint & Overfitting Manager
# --------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = float('-inf')
        self.best_path = self.save_dir / "best_checkpoint.pkl"
        self.history_path = self.save_dir / "training_history.pkl"

    def save(self, params, opt_state, step, metrics, is_best=False):
        checkpoint = {'params': params, 'opt_state': opt_state, 'step': step, 'metrics': metrics}
        latest_path = self.save_dir / f"checkpoint_{step:06d}.pkl"
        with open(latest_path, 'wb') as f: pickle.dump(checkpoint, f)
        if is_best:
            self.best_metric = metrics.get('full_result_acc', 0)
            with open(self.best_path, 'wb') as f: pickle.dump(checkpoint, f)
            print(f"  [ckpt] New best (full_acc={self.best_metric:.1f}%)")
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl"))
        while len(checkpoints) > self.max_checkpoints: checkpoints.pop(0).unlink()

    def load_best(self):
        if self.best_path.exists():
            with open(self.best_path, 'rb') as f: return pickle.load(f)
        return None
    def load_latest(self):
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl"))
        if checkpoints:
            with open(checkpoints[-1], 'rb') as f: return pickle.load(f)
        return None
    def save_history(self, history):
        with open(self.history_path, 'wb') as f: pickle.dump(history, f)
    def load_history(self):
        if self.history_path.exists():
            with open(self.history_path, 'rb') as f: return pickle.load(f)
        return None

class OverfittingDetector:
    def __init__(self, patience=20, min_delta=0.5, max_degradation=5.0, 
                 divergence_window=10, divergence_threshold=0.3):
        self.patience = patience
        self.min_delta = min_delta
        self.max_degradation = max_degradation
        self.divergence_window = divergence_window
        self.divergence_threshold = divergence_threshold
        self.best_val_acc = float('-inf')
        self.wait_count = 0
        self.stopped = False
        self.warnings = []
        self.train_losses = []
        self.val_accs = []
        self.overfitting_detected = False
    
    def check_divergence(self):
        if len(self.train_losses) < self.divergence_window: return False, ""
        recent_train = self.train_losses[-self.divergence_window:]
        recent_val = self.val_accs[-self.divergence_window:]
        train_trend = recent_train[-1] - recent_train[0]
        val_trend = recent_val[-1] - recent_val[0]
        if train_trend < -self.divergence_threshold and val_trend <= 0:
            return True, f"DIVERGENCE: train_loss {train_trend:+.3f} but val_acc {val_trend:+.1f}%"
        return False, ""
    
    def __call__(self, train_loss, val_acc):
        self.train_losses.append(train_loss)
        self.val_accs.append(val_acc)
        if val_acc > (self.best_val_acc + self.min_delta):
            self.best_val_acc = val_acc
            self.wait_count = 0
            self.overfitting_detected = False
            return False, f"improving (best={self.best_val_acc:.1f}%)"
        self.wait_count += 1
        if (self.best_val_acc - val_acc) > self.max_degradation:
            self.stopped = True
            self.overfitting_detected = True
            return True, f"DEGRADED {self.best_val_acc - val_acc:.1f}% from peak"
        is_diverging, div_desc = self.check_divergence()
        if is_diverging and self.wait_count >= self.patience // 2:
            self.warnings.append(div_desc)
            if len(self.warnings) >= 3:
                self.stopped = True
                self.overfitting_detected = True
                return True, f"SUSTAINED {div_desc}"
        if self.wait_count >= self.patience:
            self.stopped = True
            return True, f"no improvement for {self.patience} chunks"
        if is_diverging: return False, f"WARNING: {div_desc}"
        return False, f"waiting ({self.wait_count}/{self.patience})"
    
    def get_status_report(self):
        return f"Best val accuracy: {self.best_val_acc:.1f}% | Overfitting: {self.overfitting_detected}"

# --------------------------------------------------------------------
# Advanced Data Generation (Handles *, variable lengths)
# --------------------------------------------------------------------
def generate_equations(num_range, num_samples, ops=['+', '-', '*']):
    pairs = []
    for _ in range(num_samples):
        a = random.randint(*num_range)
        b = random.randint(*num_range)
        op = random.choice(ops)
        
        if op == '+': result = a + b
        elif op == '-': result = a - b
        else: result = a * b  # Multiplication
        
        # Dynamic formatting based on result size and sign
        res_str = f"{result:04d}" if result < 0 else f"{result:0d}" 
        eq = f"{a:02d}{op}{b:02d}={res_str}"
        pairs.append(eq)
    return pairs

def get_result_positions(ids):
    eq_idx = list(ids).index(tokenizer.eq_id)
    return list(range(eq_idx + 1, len(ids)))

# --------------------------------------------------------------------
# Masked Batch
# --------------------------------------------------------------------
def create_masked_batch(equations, rng_key, mask_prob=0.8):
    B = bert.CONFIG["batch_size"]
    T = bert.CONFIG["seq_len"]
    input_ids = np.zeros((B, T), dtype=np.int32)
    target_ids = np.zeros((B, T), dtype=np.int32)
    mlm_mask  = np.zeros((B, T), dtype=np.bool_)

    for i, eq in enumerate(equations):
        ids = np.array(tokenizer.encode(eq))
        if len(ids) >= T: continue
        res_pos = get_result_positions(ids)
        masked = ids.copy()
        for pos in res_pos:
            if random.random() < mask_prob:
                masked[pos] = tokenizer.mask_id
                mlm_mask[i, pos] = True
        input_ids[i, :len(ids)] = masked
        target_ids[i, :len(ids)] = ids
    return jnp.array(input_ids), jnp.array(target_ids), jnp.array(mlm_mask)

# --------------------------------------------------------------------
# Probe: Dynamic Positional Accuracy
# --------------------------------------------------------------------
def probe_accuracy_detailed(params, val_eqs, num_samples=500):
    pos_ok = {}
    pos_tot = {}
    full_ok = 0
    full_tot = 0
    key = jax.random.PRNGKey(0)
    T = bert.CONFIG["seq_len"]

    for _ in range(num_samples):
        eq = random.choice(val_eqs)
        ids = np.array(tokenizer.encode(eq))
        res_pos = get_result_positions(ids)
        res_len = len(res_pos)
        
        masked = ids.copy()
        for pos in res_pos: masked[pos] = tokenizer.mask_id

        inp = np.zeros((1, T), dtype=np.int32)
        inp[0, :len(ids)] = masked

        key, subkey = jax.random.split(key)
        logits = bert.forward(params, inp, T, bert.CONFIG["num_heads"], 
                              bert.CONFIG["num_kv_heads"], bert.CONFIG["d_model"] // bert.CONFIG["num_heads"],
                              bert.CONFIG["eps_rms"], subkey, False)

        preds = [int(jnp.argmax(logits[0, pos])) for pos in res_pos]
        targets = [int(ids[p]) for p in res_pos]
        
        all_correct = True
        for i, (p, t) in enumerate(zip(preds, targets)):
            rel_pos = res_len - 1 - i
            if rel_pos not in pos_tot: pos_tot[rel_pos] = 0
            if rel_pos not in pos_ok: pos_ok[rel_pos] = 0
            pos_tot[rel_pos] += 1
            if p == t: pos_ok[rel_pos] += 1
            else: all_correct = False
            
        full_tot += 1
        if all_correct: full_ok += 1

    def pct(ok, tot): return 100.0 * ok / tot if tot > 0 else 0.0
    names = {0: "Ones", 1: "Tens", 2: "Hunds", 3: "Thous", 4: "TenTh"}
    
    results = {f"pos_{names.get(k, k)}": pct(pos_ok.get(k, 0), pos_tot.get(k, 0)) for k in sorted(pos_tot.keys())}
    results["full_result_acc"] = pct(full_ok, full_tot)
    return results

def compute_val_loss(params, val_eqs, num_batches=20):
    key = jax.random.PRNGKey(42)
    total_loss = 0.0
    for _ in range(num_batches):
        key, subkey, dropkey = jax.random.split(key, 3)
        batch_eqs = random.choices(val_eqs, k=bert.CONFIG["batch_size"])
        inp, targ, mask = create_masked_batch(batch_eqs, subkey)
        loss = bert.mlm_loss(params, inp, targ, mask, bert.CONFIG, dropkey)
        total_loss += float(loss)
    return total_loss / num_batches

def get_lr(step, warmup_steps, base_lr, total_steps=10000):
    if step < warmup_steps: return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# --------------------------------------------------------------------
# Training (With Gradient Accumulation)
# --------------------------------------------------------------------
def train_bert(train_eqs, val_eqs, steps=10000, resume_from=None, save_dir="checkpoints/bert_math_adv"):
    key = jax.random.PRNGKey(42)
    ckpt_mgr = CheckpointManager(save_dir, max_checkpoints=5)
    overfit_detector = OverfittingDetector(patience=20)

    start_step = 0
    ckpt = ckpt_mgr.load_best() if resume_from == 'best' else ckpt_mgr.load_latest() if resume_from == 'latest' else None

    if ckpt:
        params, opt_state, start_step = ckpt['params'], ckpt['opt_state'], ckpt['step']
        print(f"Resumed from step {start_step}")
    else:
        params = bert.init_params_bert(key, bert.CONFIG["vocab_size"], bert.CONFIG["d_model"],
                                       bert.CONFIG["num_layers"], bert.CONFIG["num_heads"],
                                       bert.CONFIG["num_kv_heads"], bert.CONFIG["seq_len"])
        opt_state = bert.init_optimizer_state(params)

    auditor = SparkAuditor(max_grad_norm=bert.CONFIG["max_grad_norm"])
    history = {'train_loss': [], 'val_loss': [], 'val_full': [], 'steps': []}
    old_history = ckpt_mgr.load_history()
    if old_history:
        for k in history: history[k] = old_history.get(k, [])

    chunk_size = bert.CONFIG["chunk_size"]
    total_chunks = steps // chunk_size
    global_step = start_step
    val_results = {"full_result_acc": 0.0}
    val_loss, is_best = float('inf'), False

    print(f"\n{'='*80}")
    print(f"Advanced Training: steps {start_step}->{steps} | Task: +, -, * (up to 4 digits)")
    print(f"Model: d={bert.CONFIG['d_model']}, L={bert.CONFIG['num_layers']}, H={bert.CONFIG['num_heads']}")
    print(f"Memory: Physical Batch={bert.CONFIG['batch_size']}, Accum Steps={GRAD_ACCUM_STEPS} "
          f"(Effective Batch={bert.CONFIG['batch_size'] * GRAD_ACCUM_STEPS})")
    print(f"{'='*80}\n")

    for chunk in range(start_step // chunk_size, total_chunks):
        if overfit_detector.stopped: break
        loss_sum = 0.0
        for _ in range(chunk_size):
            key, subkey, dropkey = jax.random.split(key, 3)
            lr = get_lr(global_step, bert.CONFIG["warmup_steps"], bert.CONFIG["lr"], steps)
            batch_eqs = random.choices(train_eqs, k=bert.CONFIG["batch_size"])
            inp, targ, mask = create_masked_batch(batch_eqs, subkey)
            
            # KEY FIX: Divide loss by accumulation steps. 
            # This simulates a large batch without allocating the VRAM for one.
            loss_fn = lambda p: bert.mlm_loss(p, inp, targ, mask, bert.CONFIG, dropkey) / GRAD_ACCUM_STEPS
            loss, grads = jax.value_and_grad(loss_fn)(params)
            
            grads, _ = bert.clip_grads(grads, bert.CONFIG["max_grad_norm"])
            params, opt_state = bert.apply_adamw_update(params, grads, opt_state, lr, bert.CONFIG["adam_b1"], bert.CONFIG["adam_b2"], bert.CONFIG["adam_eps"], bert.CONFIG["weight_decay"])
            
            loss_sum += float(loss) * GRAD_ACCUM_STEPS # Scale back up for logging
            global_step += 1

        train_loss = loss_sum / chunk_size
        auditor.loss_history.append(train_loss)

        if (chunk + 1) % 5 == 0:
            val_results = probe_accuracy_detailed(params, val_eqs, 500)
            val_loss = compute_val_loss(params, val_eqs, 20)
            auditor.update_probe_scores(val_results)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_full'].append(val_results['full_result_acc'])
            history['steps'].append(global_step)

            should_stop, status = overfit_detector(train_loss, val_results['full_result_acc'])
            gap = val_loss - train_loss
            
            pos_strs = [f"{k}={v:5.1f}%" for k, v in val_results.items() if k.startswith("pos_")]
            pos_printout = " | ".join(pos_strs) if pos_strs else ""
            
            print(f"Chunk {chunk+1:3d} (step {global_step:5d}): train={train_loss:.4f} val={val_loss:.4f} gap={gap:+.4f} lr={lr:.2e}")
            print(f"         Digits: {pos_printout} | Full={val_results['full_result_acc']:5.1f}% | {status}")

            is_best = val_results['full_result_acc'] > (ckpt_mgr.best_metric + 0.5)
            if (chunk + 1) % 10 == 0:
                ckpt_mgr.save(params, opt_state, global_step, {'val_loss': val_loss, **val_results}, is_best)
                ckpt_mgr.save_history(history)
            if should_stop:
                print(f"\n{'!'*80}\nSTOPPED: {status}\n{'!'*80}")

    ckpt_mgr.save(params, opt_state, global_step, {'val_loss': val_loss, **val_results}, is_best)
    ckpt_mgr.save_history(history)
    print(f"\nDone. Best full val accuracy: {ckpt_mgr.best_metric:.1f}%")
    
    best_ckpt = ckpt_mgr.load_best()
    return best_ckpt['params'] if best_ckpt else params

# --------------------------------------------------------------------
# Interactive Test Loop
# --------------------------------------------------------------------
def interactive_loop(params):
    print("\n" + "="*60)
    print("INTERACTIVE ADVANCED MODE (Type 'quit' to exit)")
    print("Format: XX+YY, XX-YY, or XX*YY")
    print("="*60)
    
    T = bert.CONFIG["seq_len"]
    key = jax.random.PRNGKey(999)
    
    while True:
        try:
            user_input = input("\nEquation: ").strip().replace(" ", "").replace("=", "")
            if user_input.lower() in ['quit', 'exit', 'q']: break
                
            if '+' in user_input:
                a, b = map(int, user_input.split('+'))
                op = '+'
            elif '*' in user_input:
                a, b = map(int, user_input.split('*'))
                op = '*'
            elif '-' in user_input:
                parts = user_input.split('-')
                if len(parts) != 2: raise ValueError
                a, b = int(parts[0]), int(parts[1])
                op = '-'
            else:
                print("Invalid format.")
                continue
                
            if not (0 <= a <= 99 and 0 <= b <= 99):
                print("Numbers must be between 0 and 99.")
                continue
                
            result = a + b if op == '+' else (a - b if op == '-' else a * b)
            true_eq = f"{a:02d}{op}{b:02d}={result}"
            
            eq_idx = true_eq.index('=')
            masked_eq = true_eq[:eq_idx+1] + '?' * (len(true_eq) - eq_idx - 1)
            
            ids = np.array(tokenizer.encode(masked_eq))
            inp = np.zeros((1, T), dtype=np.int32)
            inp[0, :len(ids)] = ids
            
            key, subkey = jax.random.split(key)
            logits = bert.forward(params, inp, T, bert.CONFIG["num_heads"], bert.CONFIG["num_kv_heads"],
                                  bert.CONFIG["d_model"] // bert.CONFIG["num_heads"], bert.CONFIG["eps_rms"], subkey, False)
            
            pred_chars = list(masked_eq)
            for i, c in enumerate(masked_eq):
                if c == '?':
                    pred_chars[i] = tokenizer.id_to_char[int(jnp.argmax(logits[0, i]))]
                    
            pred_str = "".join(pred_chars)
            mark = '✓' if pred_str == true_eq else '✗'
            print(f"True:      {true_eq}")
            print(f"Predicted: {pred_str}  {mark}")
            
        except Exception as e:
            print(f"Error: {e}. Please use format XX+YY, XX-YY, or XX*YY")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', choices=['best', 'latest', None], default=None)
    parser.add_argument('--save-dir', default='checkpoints/bert_math_adv')
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()

    if not args.skip_train:
        print("Generating advanced data (+, -, * up to 99x99)...")
        train_eqs = generate_equations((1, 99), 100000, ops=['+', '-', '*']) 
        val_eqs   = generate_equations((1, 99), 5000, ops=['+', '-', '*'])
        print(f"Train: {len(train_eqs)} | Val: {len(val_eqs)}")
        
        ex_add = [e for e in train_eqs if '+' in e][:1][0]
        ex_sub = [e for e in train_eqs if '-' in e and '-' in e.split('=')[1]][:1][0]
        ex_mul = [e for e in train_eqs if '*' in e and len(e) > 9][:1][0]
        print(f"Examples: {ex_add}, {ex_sub}, {ex_mul}")

        params = train_bert(train_eqs, val_eqs, steps=15000, resume_from=args.resume, save_dir=args.save_dir)
    else:
        print("Loading best checkpoint...")
        ckpt_mgr = CheckpointManager(args.save_dir)
        ckpt = ckpt_mgr.load_best()
        if not ckpt:
            print("No checkpoint found!"); sys.exit(1)
        params = ckpt['params']

    interactive_loop(params)