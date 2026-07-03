# exercise_bert_math.py
"""
Improved Masked Arithmetic Solver (Bidirectional BERT)
- Larger model (d_model=256, 6 layers)
- Longer training (5000 steps)
- LR=2e-4 with warmup + cosine decay
- Masks only result digits (unambiguous task)
- Checkpointing + overfitting-aware early stopping
- Probe monitors VAL set, early stopping uses VAL metrics
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
        self.vocab_size = len(self.char_to_id)
        self.result_positions = [6, 7]

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
# Checkpoint manager
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
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'step': step,
            'metrics': metrics,
        }
        latest_path = self.save_dir / f"checkpoint_{step:06d}.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        if is_best:
            self.best_metric = metrics.get('both_correct', 0)
            with open(self.best_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  [ckpt] New best (val_both={self.best_metric:.1f}%)")
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl"))
        while len(checkpoints) > self.max_checkpoints:
            checkpoints.pop(0).unlink()

    def load_best(self):
        if self.best_path.exists():
            with open(self.best_path, 'rb') as f:
                return pickle.load(f)
        return None

    def load_latest(self):
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl"))
        if checkpoints:
            with open(checkpoints[-1], 'rb') as f:
                return pickle.load(f)
        return None

    def save_history(self, history):
        with open(self.history_path, 'wb') as f:
            pickle.dump(history, f)

    def load_history(self):
        if self.history_path.exists():
            with open(self.history_path, 'rb') as f:
                return pickle.load(f)
        return None

# --------------------------------------------------------------------
# Overfitting Detector
# --------------------------------------------------------------------
class OverfittingDetector:
    def __init__(self, 
                 patience=15, min_delta=0.5, max_degradation=5.0, 
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
        if len(self.train_losses) < self.divergence_window:
            return False, ""
        
        recent_train = self.train_losses[-self.divergence_window:]
        recent_val = self.val_accs[-self.divergence_window:]
        
        train_trend = recent_train[-1] - recent_train[0]
        val_trend = recent_val[-1] - recent_val[0]
        
        if train_trend < -self.divergence_threshold and val_trend <= 0:
            desc = (f"DIVERGENCE: train_loss {train_trend:+.3f} but "
                    f"val_acc {val_trend:+.1f}%")
            return True, desc
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
        
        degradation = self.best_val_acc - val_acc
        if degradation > self.max_degradation:
            msg = f"DEGRADED {degradation:.1f}% from peak {self.best_val_acc:.1f}%"
            self.stopped = True
            self.overfitting_detected = True
            return True, msg
        
        is_diverging, div_desc = self.check_divergence()
        if is_diverging and self.wait_count >= self.patience // 2:
            self.warnings.append(f"Chunk {len(self.train_losses)}: {div_desc}")
            if len(self.warnings) >= 3:
                self.stopped = True
                self.overfitting_detected = True
                return True, f"SUSTAINED {div_desc}"
        
        if self.wait_count >= self.patience:
            self.stopped = True
            return True, f"no improvement for {self.patience} chunks"
        
        if is_diverging:
            return False, f"WARNING: {div_desc}"
        return False, f"waiting ({self.wait_count}/{self.patience})"
    
    def get_status_report(self):
        report = [
            f"Best val accuracy: {self.best_val_acc:.1f}%",
            f"Current wait: {self.wait_count}/{self.patience}",
            f"Overfitting detected: {self.overfitting_detected}"
        ]
        if self.warnings:
            report.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings[-3:]:
                report.append(f"  - {w}")
        return "\n".join(report)

# --------------------------------------------------------------------
# Data generation: "XX+YY=ZZ" format
# --------------------------------------------------------------------
def generate_equations(num_range, num_samples, ops=['+','-']):
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
        eq = f"{a:02d}{op}{b:02d}={result:02d}"
        if len(eq) == 8 and 0 <= result <= 99:
            pairs.append(eq)
    return pairs

# --------------------------------------------------------------------
# Masked batch
# --------------------------------------------------------------------
def create_masked_batch(equations, rng_key, mask_prob=0.8):
    B = bert.CONFIG["batch_size"]
    T = bert.CONFIG["seq_len"]
    input_ids = np.zeros((B, T), dtype=np.int32)
    target_ids = np.zeros((B, T), dtype=np.int32)
    mlm_mask  = np.zeros((B, T), dtype=np.bool_)

    for i, eq in enumerate(equations):
        if len(eq) != 8: continue
        ids = np.array(tokenizer.encode(eq))
        masked = ids.copy()
        for pos in tokenizer.result_positions:
            if random.random() < mask_prob:
                masked[pos] = tokenizer.mask_id
                mlm_mask[i, pos] = True
        input_ids[i, :8] = masked
        target_ids[i, :8] = ids
    return jnp.array(input_ids), jnp.array(target_ids), jnp.array(mlm_mask)

# --------------------------------------------------------------------
# Probe: runs on VALIDATION set
# --------------------------------------------------------------------
def probe_accuracy_detailed(params, val_eqs, num_samples=300):
    correct_by_pos = {6: 0, 7: 0}
    total_by_pos = {6: 0, 7: 0}
    both_correct = 0
    total = 0
    key = jax.random.PRNGKey(0)

    for _ in range(num_samples):
        eq = random.choice(val_eqs)
        if len(eq) != 8: continue
        ids = np.array(tokenizer.encode(eq))
        masked = ids.copy()
        for pos in tokenizer.result_positions:
            masked[pos] = tokenizer.mask_id

        T = bert.CONFIG["seq_len"]
        inp = np.zeros((1, T), dtype=np.int32)
        inp[0, :8] = masked

        key, subkey = jax.random.split(key)
        logits = bert.forward(
            params, inp, T, bert.CONFIG["num_heads"], bert.CONFIG["num_kv_heads"],
            bert.CONFIG["d_model"] // bert.CONFIG["num_heads"],
            bert.CONFIG["eps_rms"], subkey, False
        )

        all_correct = True
        for pos in tokenizer.result_positions:
            pred = int(jnp.argmax(logits[0, pos]))
            total_by_pos[pos] += 1
            if pred == ids[pos]:
                correct_by_pos[pos] += 1
            else:
                all_correct = False
        total += 1
        if all_correct:
            both_correct += 1

    return {
        "tens_acc": 100.0 * correct_by_pos[6] / total_by_pos[6] if total_by_pos[6] else 0,
        "ones_acc": 100.0 * correct_by_pos[7] / total_by_pos[7] if total_by_pos[7] else 0,
        "both_correct": 100.0 * both_correct / total if total else 0,
    }

# --------------------------------------------------------------------
# Compute validation loss
# --------------------------------------------------------------------
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

# --------------------------------------------------------------------
# LR schedule
# --------------------------------------------------------------------
def get_lr(step, warmup_steps, base_lr, total_steps=5000):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# --------------------------------------------------------------------
# Training
# --------------------------------------------------------------------
def train_bert(train_eqs, val_eqs, steps=5000,
               resume_from=None, save_dir="checkpoints/bert_math"):
    key = jax.random.PRNGKey(42)
    ckpt_mgr = CheckpointManager(save_dir, max_checkpoints=5)
    overfit_detector = OverfittingDetector(
        patience=15, min_delta=0.5, max_degradation=5.0,
        divergence_window=10, divergence_threshold=0.3
    )

    start_step = 0
    if resume_from == 'best':
        ckpt = ckpt_mgr.load_best()
    elif resume_from == 'latest':
        ckpt = ckpt_mgr.load_latest()
    else:
        ckpt = None

    if ckpt:
        params = ckpt['params']
        opt_state = ckpt['opt_state']
        start_step = ckpt['step']
        print(f"Resumed from step {start_step}")
    else:
        params = bert.init_params_bert(
            key, bert.CONFIG["vocab_size"], bert.CONFIG["d_model"],
            bert.CONFIG["num_layers"], bert.CONFIG["num_heads"],
            bert.CONFIG["num_kv_heads"], bert.CONFIG["seq_len"]
        )
        opt_state = bert.init_optimizer_state(params)

    auditor = SparkAuditor(max_grad_norm=bert.CONFIG["max_grad_norm"])
    
    # FIX: Safely merge old history schema with new schema expectations
    default_history = {
        'train_loss': [], 'val_loss': [], 'val_tens_acc': [], 
        'val_ones_acc': [], 'val_both_correct': [], 'steps': []
    }
    old_history = ckpt_mgr.load_history()
    if old_history:
        for k in default_history:
            default_history[k] = old_history.get(k, [])
    history = default_history

    chunk_size = bert.CONFIG["chunk_size"]
    total_chunks = steps // chunk_size
    global_step = start_step

    # FIX: Initialize these to prevent UnboundLocalError at final save
    val_results = {"tens_acc": 0.0, "ones_acc": 0.0, "both_correct": 0.0}
    val_loss = float('inf')
    is_best = False

    print(f"\n{'='*65}")
    print(f"Training: steps {start_step}->{steps} | Save: {save_dir}")
    print(f"Overfitting detection: patience={overfit_detector.patience}, "
          f"max_degradation={overfit_detector.max_degradation}%")
    print(f"{'='*65}\n")

    for chunk in range(start_step // chunk_size, total_chunks):
        if overfit_detector.stopped:
            break

        loss_sum = 0.0
        for _ in range(chunk_size):
            key, subkey, dropkey = jax.random.split(key, 3)
            lr = get_lr(global_step, bert.CONFIG["warmup_steps"], bert.CONFIG["lr"], steps)

            batch_eqs = random.choices(train_eqs, k=bert.CONFIG["batch_size"])
            inp, targ, mask = create_masked_batch(batch_eqs, subkey)

            loss_fn = lambda p: bert.mlm_loss(p, inp, targ, mask, bert.CONFIG, dropkey)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads, _ = bert.clip_grads(grads, bert.CONFIG["max_grad_norm"])
            params, opt_state = bert.apply_adamw_update(
                params, grads, opt_state, lr,
                bert.CONFIG["adam_b1"], bert.CONFIG["adam_b2"],
                bert.CONFIG["adam_eps"], bert.CONFIG["weight_decay"]
            )
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
            history['val_tens_acc'].append(val_results['tens_acc'])
            history['val_ones_acc'].append(val_results['ones_acc'])
            history['val_both_correct'].append(val_results['both_correct'])
            history['steps'].append(global_step)

            should_stop, status = overfit_detector(train_loss, val_results['both_correct'])
            lr_now = get_lr(global_step, bert.CONFIG["warmup_steps"], bert.CONFIG["lr"], steps)
            loss_gap = val_loss - train_loss
            
            print(f"Chunk {chunk+1:3d} (step {global_step:5d}): "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"gap={loss_gap:+.4f} lr={lr_now:.2e}")
            print(f"         Val: tens={val_results['tens_acc']:5.1f}% "
                  f"ones={val_results['ones_acc']:5.1f}% "
                  f"both={val_results['both_correct']:5.1f}% | {status}")

            is_best = val_results['both_correct'] > (ckpt_mgr.best_metric + 0.5)

            if (chunk + 1) % 10 == 0:
                ckpt_mgr.save(params, opt_state, global_step,
                              {'val_loss': val_loss, **val_results}, is_best)
                ckpt_mgr.save_history(history)

            if should_stop:
                print(f"\n{'!'*65}")
                print(f"STOPPED: {status}")
                print(overfit_detector.get_status_report())
                print(f"{'!'*65}")

    # Final save
    ckpt_mgr.save(params, opt_state, global_step,
                  {'val_loss': val_loss, **val_results}, is_best)
    ckpt_mgr.save_history(history)

    print(f"\n{'='*65}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*65}")
    print(overfit_detector.get_status_report())
    print(f"\nBest val accuracy: {ckpt_mgr.best_metric:.1f}%")
    print(f"Checkpoints: {save_dir}")

    best_ckpt = ckpt_mgr.load_best()
    return best_ckpt['params'] if best_ckpt else params

# --------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------
def demo_predictions(params, val_eqs, num_examples=10):
    print("\n" + "=" * 60)
    print("PREDICTION DEMO (using best checkpoint)")
    print("=" * 60)
    key = jax.random.PRNGKey(42)
    T = bert.CONFIG["seq_len"]

    for eq in random.sample(val_eqs, min(num_examples, len(val_eqs))):
        if len(eq) != 8: continue
        ids = np.array(tokenizer.encode(eq))
        masked = ids.copy()
        for pos in tokenizer.result_positions:
            masked[pos] = tokenizer.mask_id

        inp = np.zeros((1, T), dtype=np.int32)
        inp[0, :8] = masked

        key, subkey = jax.random.split(key)
        logits = bert.forward(
            params, inp, T, bert.CONFIG["num_heads"], bert.CONFIG["num_kv_heads"],
            bert.CONFIG["d_model"] // bert.CONFIG["num_heads"],
            bert.CONFIG["eps_rms"], subkey, False
        )

        print(f"\nInput:    {tokenizer.decode(masked)}")
        print(f"Target:   {eq}")
        pred_str = list(masked)
        for pos in tokenizer.result_positions:
            top5 = jnp.argsort(logits[0, pos])[-5:][::-1]
            pred = int(top5[0])
            pred_str[pos] = pred
            top5_chars = [tokenizer.id_to_char[int(i)] for i in top5]
            mark = '✓' if pred == ids[pos] else '✗'
            print(f"  Pos {pos}: pred='{tokenizer.id_to_char[pred]}' "
                  f"(top5: {''.join(top5_chars)}) {mark}")
        print(f"Predicted: {tokenizer.decode(pred_str)}")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', choices=['best', 'latest', None], default=None)
    parser.add_argument('--save-dir', default='checkpoints/bert_math')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    print("Generating data...")
    train_eqs = generate_equations((1, 50), 50000, ops=['+','-'])
    val_eqs   = generate_equations((1, 50), 5000, ops=['+','-'])
    print(f"Train: {len(train_eqs)} | Val: {len(val_eqs)} | Example: {train_eqs[0]}")

    params = train_bert(train_eqs, val_eqs, steps=5000,
                        resume_from=args.resume, save_dir=args.save_dir)

    if args.demo:
        demo_predictions(params, val_eqs)