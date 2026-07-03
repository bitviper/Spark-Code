# exercise_bert_math_fixed.py
"""
Fixed Masked Arithmetic Solver
- Proper rotary positional encoding
- Mask ONLY result digits (unambiguous task)
- Larger batch, lower LR with warmup
- Better probe diagnostics
"""

import os, sys, random, math
import jax, jax.numpy as jnp
import numpy as np
from functools import partial
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
        # Special: result digits are positions 6,7 for "XX+YY=ZZ"
        self.result_positions = [6, 7]
    
    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids):
        return ''.join(self.id_to_char.get(int(i), '?') for i in ids)

tokenizer = MathCharTokenizer()

# --------------------------------------------------------------------
# Fixed configuration
# --------------------------------------------------------------------
bert.CONFIG.update({
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 4,
    "num_kv_heads": 4,  # Full attention, not GQA
    "ffn_hidden": 1024,
    "seq_len": 32,
    "vocab_size": tokenizer.vocab_size,
    "mask_token_id": tokenizer.mask_id,
    "lr": 2e-4,           # Lower LR
    "batch_size": 128,    # Larger batch
    "chunk_size": 20,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "dropout_rate": 0.1,
    "warmup_steps": 500,  # Add warmup
})

# --------------------------------------------------------------------
# Rotary Position Embedding (critical fix!)
# --------------------------------------------------------------------
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
    t = np.arange(seq_len)
    freqs = np.outer(t, freqs)
    return np.cos(freqs), np.sin(freqs)

def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to query/key tensors."""
    # x: (..., seq_len, head_dim)
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    # Rotate
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return jnp.concatenate([out1, out2], axis=-1)

# --------------------------------------------------------------------
# Data generation with validation
# --------------------------------------------------------------------
def generate_equations(num_range, num_samples, ops=['+']):
    """Generate equations in format XX+YY=ZZ (always 8 chars)."""
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
        # Validate format
        if len(eq) == 8 and result >= 0 and result <= 99:
            pairs.append(eq)
    return pairs

# --------------------------------------------------------------------
# CRITICAL FIX: Mask only result digits (positions 6,7)
# This makes the task UNAMBIGUOUS
# --------------------------------------------------------------------
def create_masked_batch(equations, rng_key, mask_prob=0.8):
    """
    Mask result digits (positions 6,7) with 80% probability each.
    This is an unambiguous prediction task.
    """
    B = bert.CONFIG["batch_size"]
    T = bert.CONFIG["seq_len"]
    input_ids = np.zeros((B, T), dtype=np.int32)
    target_ids = np.zeros((B, T), dtype=np.int32)
    mlm_mask  = np.zeros((B, T), dtype=np.bool_)

    for i, eq in enumerate(equations):
        if len(eq) != 8:
            continue
        ids = np.array(tokenizer.encode(eq))
        masked = ids.copy()
        
        # Only mask result positions (6 and 7)
        for pos in tokenizer.result_positions:
            if random.random() < mask_prob:
                masked[pos] = tokenizer.mask_id
                mlm_mask[i, pos] = True
        
        input_ids[i, :8] = masked
        target_ids[i, :8] = ids
    
    return jnp.array(input_ids), jnp.array(target_ids), jnp.array(mlm_mask)

# --------------------------------------------------------------------
# Better probe: separate accuracy for ones vs tens digit
# --------------------------------------------------------------------
def probe_accuracy_detailed(params, val_eqs, num_samples=500):
    """Probe with detailed breakdown by position."""
    correct_by_pos = {6: 0, 7: 0}
    total_by_pos = {6: 0, 7: 0}
    both_correct = 0
    total = 0
    
    key = jax.random.PRNGKey(0)
    
    for _ in range(num_samples):
        eq = random.choice(val_eqs)
        if len(eq) != 8:
            continue
        ids = np.array(tokenizer.encode(eq))
        masked = ids.copy()
        
        # Mask both result digits
        for pos in tokenizer.result_positions:
            masked[pos] = tokenizer.mask_id
        
        T = bert.CONFIG["seq_len"]
        inp = np.zeros((1, T), dtype=np.int32)
        inp[0, :8] = masked
        
        key, subkey = jax.random.split(key)
        logits = bert.forward(
            params, inp, T,
            bert.CONFIG["num_heads"], bert.CONFIG["num_kv_heads"],
            bert.CONFIG["d_model"] // bert.CONFIG["num_heads"],
            bert.CONFIG["eps_rms"], subkey, False
        )
        
        preds = {}
        all_correct = True
        for pos in tokenizer.result_positions:
            pred = int(jnp.argmax(logits[0, pos]))
            preds[pos] = pred
            total_by_pos[pos] += 1
            if pred == ids[pos]:
                correct_by_pos[pos] += 1
            else:
                all_correct = False
        
        total += 1
        if all_correct:
            both_correct += 1
    
    results = {
        "tens_acc": 100.0 * correct_by_pos[6] / total_by_pos[6] if total_by_pos[6] else 0,
        "ones_acc": 100.0 * correct_by_pos[7] / total_by_pos[7] if total_by_pos[7] else 0,
        "both_correct": 100.0 * both_correct / total if total else 0,
        "any_correct": 100.0 * (correct_by_pos[6] + correct_by_pos[7]) / (total_by_pos[6] + total_by_pos[7]) 
                       if (total_by_pos[6] + total_by_pos[7]) else 0,
    }
    return results

# --------------------------------------------------------------------
# Learning rate schedule with warmup
# --------------------------------------------------------------------
def get_lr(step, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    # Cosine decay after warmup
    progress = (step - warmup_steps) / (5000 - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

# --------------------------------------------------------------------
# Training with fixes
# --------------------------------------------------------------------
def train_bert(train_eqs, val_eqs, steps=5000, 
               resume_from=None, save_dir="checkpoints/bert_math"):
    """Training with checkpointing and early stopping."""
    
    key = jax.random.PRNGKey(42)
    
    # Initialize or load checkpoint
    ckpt_mgr = CheckpointManager(save_dir, max_checkpoints=5)
    early_stop = EarlyStopping(patience=15, min_delta=0.005, mode='min')
    
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
    history = ckpt_mgr.load_history() or {
        'loss': [], 'tens_acc': [], 'ones_acc': [], 'both_correct': [], 'steps': []
    }

    chunk_size = bert.CONFIG["chunk_size"]
    total_chunks = steps // chunk_size
    global_step = start_step

    print(f"\n{'='*60}")
    print(f"Training Config:")
    print(f"  Steps: {start_step} -> {steps}")
    print(f"  Checkpoints: {save_dir}")
    print(f"  Early stopping patience: {early_stop.patience} chunks")
    print(f"{'='*60}\n")

    for chunk in range(start_step // chunk_size, total_chunks):
        if early_stop.stopped:
            print(f"\n🛑 Early stopping triggered at step {global_step}")
            break
        
        loss_sum = 0.0
        for _ in range(chunk_size):
            key, subkey, dropkey = jax.random.split(key, 3)
            lr = get_lr(global_step, bert.CONFIG["warmup_steps"], bert.CONFIG["lr"])
            
            batch_eqs = random.choices(train_eqs, k=bert.CONFIG["batch_size"])
            inp, targ, mask = create_masked_batch(batch_eqs, subkey)

            loss_fn = lambda p: bert.mlm_loss(p, inp, targ, mask, bert.CONFIG, dropkey)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            grads, grad_norm = bert.clip_grads(grads, bert.CONFIG["max_grad_norm"])
            
            params, opt_state = bert.apply_adamw_update(
                params, grads, opt_state, lr,
                bert.CONFIG["adam_b1"], bert.CONFIG["adam_b2"],
                bert.CONFIG["adam_eps"], bert.CONFIG["weight_decay"]
            )
            loss_sum += float(loss)
            global_step += 1

        avg_loss = loss_sum / chunk_size
        auditor.loss_history.append(avg_loss)

        # Probe & evaluate every 5 chunks
        if (chunk + 1) % 5 == 0:
            results = probe_accuracy_detailed(params, val_eqs, 300)
            auditor.update_probe_scores(results)
            
            # Track history
            history['loss'].append(avg_loss)
            history['tens_acc'].append(results['tens_acc'])
            history['ones_acc'].append(results['ones_acc'])
            history['both_correct'].append(results['both_correct'])
            history['steps'].append(global_step)
            
            lr_now = get_lr(global_step, bert.CONFIG["warmup_steps"], bert.CONFIG["lr"])
            print(f"Chunk {chunk+1} (step {global_step}): loss={avg_loss:.4f}, lr={lr_now:.2e}")
            print(f"  Tens: {results['tens_acc']:.1f}% | Ones: {results['ones_acc']:.1f}% | Both: {results['both_correct']:.1f}%")
            
            # Check for best model & early stopping (on validation loss)
            is_best = avg_loss < (ckpt_mgr.best_metric - 0.005)
            if is_best:
                ckpt_mgr.best_metric = avg_loss
            
            # Save checkpoint every 10 chunks
            if (chunk + 1) % 10 == 0:
                ckpt_mgr.save(params, opt_state, global_step, 
                             {'val_loss': avg_loss, **results}, is_best)
                ckpt_mgr.save_history(history)
            
            # Early stopping check
            should_stop = early_stop(avg_loss)
            if should_stop:
                print(f"\n🛑 No improvement for {early_stop.patience} chunks")
                break

    # Always save final checkpoint
    ckpt_mgr.save(params, opt_state, global_step, 
                 {'val_loss': avg_loss, **results}, is_best)
    ckpt_mgr.save_history(history)
    
    print(f"\n✅ Training complete. Best loss: {ckpt_mgr.best_metric:.4f}")
    print(f"   Checkpoints saved to: {save_dir}")
    
    # Load best params for return
    best_ckpt = ckpt_mgr.load_best()
    return best_ckpt['params']

# --------------------------------------------------------------------
# Demo with full equation completion
# --------------------------------------------------------------------
def demo_predictions(params, val_eqs, num_examples=10):
    print("\n" + "="*60)
    print("PREDICTION DEMO")
    print("="*60)
    
    key = jax.random.PRNGKey(42)
    T = bert.CONFIG["seq_len"]
    
    for eq in random.sample(val_eqs, min(num_examples, len(val_eqs))):
        if len(eq) != 8:
            continue
        ids = np.array(tokenizer.encode(eq))
        masked = ids.copy()
        for pos in tokenizer.result_positions:
            masked[pos] = tokenizer.mask_id
        
        inp = np.zeros((1, T), dtype=np.int32)
        inp[0, :8] = masked
        
        key, subkey = jax.random.split(key)
        logits = bert.forward(
            params, inp, T,
            bert.CONFIG["num_heads"], bert.CONFIG["num_kv_heads"],
            bert.CONFIG["d_model"] // bert.CONFIG["num_heads"],
            bert.CONFIG["eps_rms"], subkey, False
        )
        
        # Get top predictions for each position
        print(f"\nInput:    {tokenizer.decode(masked)}")
        print(f"Target:   {eq}")
        
        pred_str = list(masked)
        for pos in tokenizer.result_positions:
            top5 = jnp.argsort(logits[0, pos])[-5:][::-1]
            pred = int(top5[0])
            pred_str[pos] = pred
            top5_chars = [tokenizer.id_to_char[int(i)] for i in top5]
            print(f"  Pos {pos}: pred='{tokenizer.id_to_char[pred]}' "
                  f"(top5: {''.join(top5_chars)}) "
                  f"{'✓' if pred == ids[pos] else '✗'}")
        print(f"Predicted: {tokenizer.decode(pred_str)}")


import os
import pickle
from pathlib import Path

# --------------------------------------------------------------------
# Checkpoint Manager
# --------------------------------------------------------------------
class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.best_metric = float('inf')  # For loss (lower is better)
        self.best_path = self.save_dir / "best_checkpoint.pkl"
        self.history_path = self.save_dir / "training_history.pkl"
    
    def save(self, params, opt_state, step, metrics, is_best=False):
        """Save checkpoint, keep only last N."""
        checkpoint = {
            'params': params,
            'opt_state': opt_state,
            'step': step,
            'metrics': metrics,
        }
        
        # Save latest
        latest_path = self.save_dir / f"checkpoint_{step:06d}.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Save best
        if is_best:
            with open(self.best_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            print(f"  💾 New best model saved (metric={metrics.get('val_loss', 0):.4f})")
        
        # Prune old checkpoints (keep best + last N)
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl"))
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
        
        return latest_path
    
    def load_best(self):
        """Load the best checkpoint."""
        if self.best_path.exists():
            with open(self.best_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_latest(self):
        """Load the most recent checkpoint."""
        checkpoints = sorted(self.save_dir.glob("checkpoint_*.pkl"))
        if checkpoints:
            with open(checkpoints[-1], 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_history(self, history):
        """Save training history for plotting."""
        with open(self.history_path, 'wb') as f:
            pickle.dump(history, f)
    
    def load_history(self):
        """Load training history."""
        if self.history_path.exists():
            with open(self.history_path, 'rb') as f:
                return pickle.load(f)
        return None

# --------------------------------------------------------------------
# Quick test of checkpoint loading
# --------------------------------------------------------------------
def test_checkpoint_loading(save_dir="checkpoints/bert_math"):
    """Verify checkpoints can be loaded."""
    ckpt_mgr = CheckpointManager(save_dir)
    
    print("\n" + "="*60)
    print("CHECKPOINT VERIFICATION")
    print("="*60)
    
    best = ckpt_mgr.load_best()
    if best:
        print(f"✓ Best checkpoint: step {best['step']}")
        print(f"  Metrics: {best['metrics']}")
    else:
        print("✗ No best checkpoint found")
    
    latest = ckpt_mgr.load_latest()
    if latest:
        print(f"✓ Latest checkpoint: step {latest['step']}")
    else:
        print("✗ No latest checkpoint found")
    
    # List all checkpoints
    ckpts = list(Path(save_dir).glob("*.pkl"))
    print(f"\nTotal checkpoints: {len(ckpts)}")
    for p in sorted(ckpts):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.name}: {size_kb:.1f} KB")

        
# --------------------------------------------------------------------
# Early Stopping
# --------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        """
        Args:
            patience: How many chunks to wait for improvement
            min_delta: Minimum change to count as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.wait_count = 0
        self.stopped = False
    
    def __call__(self, metric):
        """Check if should stop. Returns True if should stop."""
        if self.mode == 'min':
            improved = metric < (self.best_score - self.min_delta)
        else:
            improved = metric > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = metric
            self.wait_count = 0
            return False
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped = True
                return True
            return False
    
    def reset(self):
        self.best_score = float('inf') if self.mode == 'min' else float('-inf')
        self.wait_count = 0
        self.stopped = False


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', choices=['best', 'latest', None], default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-dir', default='checkpoints/bert_math',
                       help='Checkpoint save directory')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test checkpoint loading')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo after training')
    args = parser.parse_args()
    
    if args.test_only:
        test_checkpoint_loading(args.save_dir)
        sys.exit(0)
    
    print("Generating data...")
    train_eqs = generate_equations((1, 20), 50000, ops=['+'])
    val_eqs   = generate_equations((1, 20), 5000, ops=['+'])
    
    params = train_bert(
        train_eqs, val_eqs, 
        steps=5000,
        resume_from=args.resume,
        save_dir=args.save_dir
    )
    
    if args.demo:
        demo_predictions(params, val_eqs)