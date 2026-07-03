# exercise_bert_math.py
"""
Masked Arithmetic Solver (Bidirectional BERT)
- Handles XX+YY=ZZ and XX-YY=-ZZ (negative numbers)
- Detailed probe: sign, tens, ones, and full accuracy
- Checkpointing + overfitting-aware early stopping
- Interactive test loop
"""

import os, sys, random, math, pickle
import jax, jax.numpy as jnp
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import spark_bert_f as bert
from spark_auditor_f import SparkAuditor

# --------------------------------------------------------------------
# Character tokenizer
# --------------------------------------------------------------------
class MathCharTokenizer:
    def __init__(self):
        chars = "0123456789+-= ?"
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
# Configuration
# --------------------------------------------------------------------
bert.CONFIG.update({
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 4,
    "num_kv_heads": 4,
    "ffn_hidden": 1024,
    "seq_len": 32,
    "vocab_size": tokenizer.vocab_size,
    "mask_token_id": tokenizer.mask_id,
    "lr": 2e-4,
    "batch_size": 128,
    "chunk_size": 20,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "dropout_rate": 0.1,
    "warmup_steps": 500,
})

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
    def __init__(self, patience=15, min_delta=0.5, max_degradation=5.0, 
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
        report = [f"Best val accuracy: {self.best_val_acc:.1f}%", f"Overfitting detected: {self.overfitting_detected}"]
        if self.warnings: report.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(report)

# --------------------------------------------------------------------
# Data Generation: Handles +, -, and Negatives
# --------------------------------------------------------------------
def generate_equations(num_range, num_samples, ops=['+', '-'], allow_negatives=True):
    pairs = []
    for _ in range(num_samples):
        a = random.randint(*num_range)
        b = random.randint(*num_range)
        op = random.choice(ops)
        
        result = a + b if op == '+' else a - b
        
        if not allow_negatives and result < 0:
            a, b = b, a
            result = a - b
            
        # Dynamic formatting: negatives get 3 chars (-05), positives get 2 (15)
        if result < 0:
            eq = f"{a:02d}{op}{b:02d}={result:03d}" # e.g. 05-10=-05
        else:
            eq = f"{a:02d}{op}{b:02d}={result:02d}" # e.g. 05+10=15
        pairs.append(eq)
    return pairs

def get_result_positions(ids):
    """Dynamically find where the result starts (after '=')"""
    eq_idx = list(ids).index(tokenizer.eq_id)
    return list(range(eq_idx + 1, len(ids)))

# --------------------------------------------------------------------
# Dynamic Masked Batch
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
# Probe: Detailed logical position accuracy
# --------------------------------------------------------------------
def probe_accuracy_detailed(params, val_eqs, num_samples=300):
    counts = {
        'sign_ok': 0, 'sign_tot': 0,
        'tens_ok': 0, 'tens_tot': 0, 
        'ones_ok': 0, 'ones_tot': 0, 
        'full_ok': 0, 'full_tot': 0
    }
    key = jax.random.PRNGKey(0)
    T = bert.CONFIG["seq_len"]

    for _ in range(num_samples):
        eq = random.choice(val_eqs)
        ids = np.array(tokenizer.encode(eq))
        res_pos = get_result_positions(ids)
        
        masked = ids.copy()
        for pos in res_pos: masked[pos] = tokenizer.mask_id

        inp = np.zeros((1, T), dtype=np.int32)
        inp[0, :len(ids)] = masked

        key, subkey = jax.random.split(key)
        logits = bert.forward(params, inp, T, bert.CONFIG["num_heads"], 
                              bert.CONFIG["num_kv_heads"], bert.CONFIG["d_model"] // bert.CONFIG["num_heads"],
                              bert.CONFIG["eps_rms"], subkey, False)

        preds = [int(jnp.argmax(logits[0, pos])) for pos in range(len(eq))]
        
        # Determine logical mapping
        is_neg = (len(res_pos) == 3) # Negative results are 3 chars (-05), positive are 2 (15)
        
        if is_neg:
            sign_abs, tens_abs, ones_abs = res_pos[0], res_pos[1], res_pos[2]
            counts['sign_tot'] += 1
            if preds[sign_abs] == ids[sign_abs]: counts['sign_ok'] += 1
        else:
            tens_abs, ones_abs = res_pos[0], res_pos[1]
            
        counts['tens_tot'] += 1
        if preds[tens_abs] == ids[tens_abs]: counts['tens_ok'] += 1
        
        counts['ones_tot'] += 1
        if preds[ones_abs] == ids[ones_abs]: counts['ones_ok'] += 1
        
        # Full result correct
        all_correct = all(preds[p] == ids[p] for p in res_pos)
        counts['full_tot'] += 1
        if all_correct: counts['full_ok'] += 1

    def pct(ok, tot): return 100.0 * ok / tot if tot > 0 else 0.0

    return {
        "sign_acc": pct(counts['sign_ok'], counts['sign_tot']),
        "tens_acc": pct(counts['tens_ok'], counts['tens_tot']),
        "ones_acc": pct(counts['ones_ok'], counts['ones_tot']),
        "full_result_acc": pct(counts['full_ok'], counts['full_tot']),
    }

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

def get_lr(step, warmup_steps, base_lr, total_steps=5000):
    if step < warmup_steps: return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
def train_bert(train_eqs, val_eqs, steps=5000, resume_from=None, save_dir="checkpoints/bert_math"):
    key = jax.random.PRNGKey(42)
    ckpt_mgr = CheckpointManager(save_dir, max_checkpoints=5)
    overfit_detector = OverfittingDetector(patience=15, min_delta=0.5, max_degradation=5.0)

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
    default_history = {'train_loss': [], 'val_loss': [], 'val_sign': [], 'val_tens': [], 'val_ones': [], 'val_full': [], 'steps': []}
    old_history = ckpt_mgr.load_history()
    if old_history:
        for k in default_history: default_history[k] = old_history.get(k, [])
    history = default_history

    chunk_size = bert.CONFIG["chunk_size"]
    total_chunks = steps // chunk_size
    global_step = start_step
    val_results = {"sign_acc": 0.0, "tens_acc": 0.0, "ones_acc": 0.0, "full_result_acc": 0.0}
    val_loss, is_best = float('inf'), False

    print(f"\n{'='*75}")
    print(f"Training: steps {start_step}->{steps} | Task: +, - & Negatives")
    print(f"{'='*75}\n")

    for chunk in range(start_step // chunk_size, total_chunks):
        if overfit_detector.stopped: break
        loss_sum = 0.0
        for _ in range(chunk_size):
            key, subkey, dropkey = jax.random.split(key, 3)
            lr = get_lr(global_step, bert.CONFIG["warmup_steps"], bert.CONFIG["lr"], steps)
            batch_eqs = random.choices(train_eqs, k=bert.CONFIG["batch_size"])
            inp, targ, mask = create_masked_batch(batch_eqs, subkey)
            loss_fn = lambda p: bert.mlm_loss(p, inp, targ, mask, bert.CONFIG, dropkey)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads, _ = bert.clip_grads(grads, bert.CONFIG["max_grad_norm"])
            params, opt_state = bert.apply_adamw_update(params, grads, opt_state, lr, bert.CONFIG["adam_b1"], bert.CONFIG["adam_b2"], bert.CONFIG["adam_eps"], bert.CONFIG["weight_decay"])
            loss_sum += float(loss)
            global_step += 1

        train_loss = loss_sum / chunk_size
        auditor.loss_history.append(train_loss)

        if (chunk + 1) % 5 == 0:
            val_results = probe_accuracy_detailed(params, val_eqs, 300)
            val_loss = compute_val_loss(params, val_eqs, 20)
            auditor.update_probe_scores(val_results)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_sign'].append(val_results['sign_acc'])
            history['val_tens'].append(val_results['tens_acc'])
            history['val_ones'].append(val_results['ones_acc'])
            history['val_full'].append(val_results['full_result_acc'])
            history['steps'].append(global_step)

            should_stop, status = overfit_detector(train_loss, val_results['full_result_acc'])
            gap = val_loss - train_loss
            
            print(f"Chunk {chunk+1:3d} (step {global_step:5d}): train={train_loss:.4f} val={val_loss:.4f} gap={gap:+.4f} lr={get_lr(global_step, bert.CONFIG['warmup_steps'], bert.CONFIG['lr'], steps):.2e}")
            print(f"         Val: sign={val_results['sign_acc']:5.1f}% | tens={val_results['tens_acc']:5.1f}% | ones={val_results['ones_acc']:5.1f}% | full={val_results['full_result_acc']:5.1f}% | {status}")

            is_best = val_results['full_result_acc'] > (ckpt_mgr.best_metric + 0.5)
            if (chunk + 1) % 10 == 0:
                ckpt_mgr.save(params, opt_state, global_step, {'val_loss': val_loss, **val_results}, is_best)
                ckpt_mgr.save_history(history)
            if should_stop:
                print(f"\n{'!'*75}\nSTOPPED: {status}\n{'!'*75}")

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
    print("INTERACTIVE MODE (Type 'quit' to exit)")
    print("Format: XX+YY or XX-YY (e.g. 12+05 or 45-50)")
    print("="*60)
    
    T = bert.CONFIG["seq_len"]
    key = jax.random.PRNGKey(999)
    
    while True:
        try:
            user_input = input("\nEquation: ").strip().replace(" ", "")
            if user_input.lower() in ['quit', 'exit', 'q']: break
                
            if '+' in user_input:
                a, b = map(int, user_input.split('+'))
                op = '+'
            elif '-' in user_input:
                a, b = map(int, user_input.split('-'))
                op = '-'
            else:
                print("Invalid format. Use XX+YY or XX-YY")
                continue
                
            if not (0 <= a <= 99 and 0 <= b <= 99):
                print("Numbers must be between 0 and 99.")
                continue
                
            result = a + b if op == '+' else a - b
            true_eq = f"{a:02d}{op}{b:02d}={result:03d}" if result < 0 else f"{a:02d}{op}{b:02d}={result:02d}"
            
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
            print(f"Error: {e}. Please use format XX+YY or XX-YY")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', choices=['best', 'latest', None], default=None)
    parser.add_argument('--save-dir', default='checkpoints/bert_math')
    parser.add_argument('--skip-train', action='store_true', help='Go straight to interactive loop')
    args = parser.parse_args()

    if not args.skip_train:
        print("Generating data (+, -, and Negatives)...")
        train_eqs = generate_equations((1, 99), 50000, ops=['+', '-'], allow_negatives=True)
        val_eqs   = generate_equations((1, 99), 5000, ops=['+', '-'], allow_negatives=True)
        print(f"Train: {len(train_eqs)} | Val: {len(val_eqs)}")
        print(f"Examples: {train_eqs[0]}, {train_eqs[1]}, {train_eqs[2]}")

        params = train_bert(train_eqs, val_eqs, steps=5000, resume_from=args.resume, save_dir=args.save_dir)
    else:
        print("Loading best checkpoint for interactive mode...")
        ckpt_mgr = CheckpointManager(args.save_dir)
        ckpt = ckpt_mgr.load_best()
        if not ckpt:
            print("No checkpoint found! Train first.")
            sys.exit(1)
        params = ckpt['params']

    interactive_loop(params)