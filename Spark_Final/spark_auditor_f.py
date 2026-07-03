# spark_auditor_f.py
# Training auditor with gradient monitoring, loss anomaly detection,
# entropy collapse detection, and safe checkpoint rollback.
#
# Critical fixes:
# - Rollback uses jax.device_put to avoid recompilation
# - Loss threshold now relative to starting loss
# - LR throttle returns a boolean flag instead of new LR
# - Heatmap correctly handles list of attention weights from layers

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os

class SparkAuditor:
    def __init__(self, max_grad_norm=1.0, lr_throttle_threshold=5.0, anomaly_threshold=1.5, vocab_size=5000):
        self.max_grad_norm = max_grad_norm
        self.lr_throttle_threshold = lr_throttle_threshold
        self.anomaly_threshold = anomaly_threshold
        self.vocab_size = vocab_size
        
        self.loss_history = []
        self.grad_history = []
        self.entropy_history = []
        self.val_loss_history = []
        
        self.overfit_patience = 3   
        self.probe_scores = {}       
        self.probe_latest = {}       
        self.probe_count = 0

    def update_val_loss(self, val_loss):
        self.val_loss_history.append(float(val_loss))

    def check_overfitting(self):
        if len(self.val_loss_history) < self.overfit_patience + 1:
            return False, ""
        last = self.val_loss_history[-self.overfit_patience - 1:]
        if all(last[i] < last[i+1] for i in range(len(last) - 1)):
            return True, "⚠️ OVERFITTING: Validation loss rising for 3 consecutive checks. Stop training or increase dropout / weight decay."
        return False, ""
        
    def inspect_grad_norm(self, grad_norm, current_lr):
        """Monitor a pre-computed global gradient norm (scalar).
        Returns (should_throttle_lr, is_vanishing, grad_norm)."""
        g = float(grad_norm)
        self.grad_history.append(g)

        should_throttle = g > self.lr_throttle_threshold
        is_vanishing = g < 1e-6
        return should_throttle, is_vanishing, g

    def inspect_gradients(self, grads, current_lr):
        """Scans a gradient pytree and tracks global norm."""
        squared_tree = jax.tree_util.tree_map(lambda g: jnp.sum(g ** 2), grads)
        global_grad_norm = float(jnp.sqrt(sum(jax.tree_util.tree_leaves(squared_tree))))
        return self.inspect_grad_norm(global_grad_norm, current_lr)

    def monitor_loss_anomaly(self, current_loss):
        c_loss = float(current_loss)
        self.loss_history.append(c_loss)
        if len(self.loss_history) < 15:
            return False
        if np.isnan(c_loss) or np.isinf(c_loss):
            return True
        recent_avg = sum(self.loss_history[-15:-1]) / 14
        if c_loss > recent_avg * self.anomaly_threshold:
            return True
        return False

    def monitor_entropy(self, current_entropy):
        c_ent = float(current_entropy)
        self.entropy_history.append(c_ent)
        if len(self.entropy_history) < 20:
            return False, "Nominal"
        if c_ent < 0.15:
            return True, "⚠️ [DIAGNOSIS] FATAL MEMORIZATION: Entropy collapsed (<0.15)."
        return False, "Nominal"

    def diagnose_bottleneck(self):
        if len(self.loss_history) < 20 or len(self.entropy_history) < 20:
            return "📋 [Auditor] Gathering baseline data telemetry..."

        recent_losses = self.loss_history[-20:]
        recent_grads = self.grad_history[-20:]
        recent_entropies = self.entropy_history[-20:]

        loss_variance = float(np.var(recent_losses))
        avg_grad = float(np.mean(recent_grads))
        avg_entropy = float(np.mean(recent_entropies))

        directions = np.diff(recent_losses)
        sign_changes = np.sum(directions[:-1] * directions[1:] < 0)

        # 1. LR too high (oscillation)
        if sign_changes > 12 and loss_variance > 0.1:
            return "⚠️ [DIAGNOSIS] LEARNING RATE TOO HIGH: Loss is oscillating violently."

        # 2. LR too low
        loss_delta = abs(recent_losses[0] - recent_losses[-1])
        if loss_delta < 1e-4 and avg_grad > 1e-3 and loss_variance < 1e-6:
            return "⚠️ [DIAGNOSIS] LEARNING RATE TOO LOW: Gradients active, loss frozen."

        # 3. Capacity or saturation (relative to starting loss)
        if loss_variance < 1e-4 and loss_delta < 0.05:
            initial_loss = self.loss_history[0]
            if recent_losses[-1] > initial_loss * 0.75:
                return "⚠️ [DIAGNOSIS] CAPACITY LIMIT: Loss flatlining high."
            elif avg_entropy < 0.15:
                return "⚠️ [DIAGNOSIS] DATA SATURATION: Memorisation absolute."

        # 4. Vanishing gradients
        if avg_grad < 1e-5:
            return "⚠️ [DIAGNOSIS] OPTIMIZATION LOCK: Vanishing gradients."

        return f"✅ [DIAGNOSIS] HEALTHY: Converging smoothly. (Entropy: {avg_entropy:.2f})"

    def render_attention_heatmap(self, attention_weights, tokens=None, max_seq_display=32):
        # FIX: Handle list of arrays (from multiple layers) by stacking and averaging
        if isinstance(attention_weights, list):
            stacked_weights = jnp.stack(attention_weights)  # Shape: (num_layers, B, H, T, T)
            mean_weights = np.array(jnp.mean(stacked_weights, axis=(0, 1, 2)))
        else:
            mean_weights = np.array(jnp.mean(attention_weights, axis=(0, 1)))
            
        T = min(mean_weights.shape[0], max_seq_display)

        print("\n--- [AUDITOR] LIVE CAUSAL ATTENTION MATRIX MAP ---")
        display_tokens = [f"T{i:02d}" for i in range(T)] if tokens is None else [str(t)[:4] for t in tokens[:T]]

        for i in range(len(display_tokens)):
            row_str = f"  {display_tokens[i]:>3} | "
            for j in range(T):
                w = mean_weights[i, j]
                if w < 0.05:     row_str += "."
                elif w < 0.20:   row_str += "*"
                elif w < 0.50:   row_str += "#"
                else:            row_str += "█"
            print(row_str)
        print("-" * (T + 8) + "\n")

    def checkpoint_rollback_state(self, params, opt_state, step, filename="spark_3_rollback.pkl"):
        flat_data = {
            "params": jax.device_get(params),
            "opt_state": jax.device_get(opt_state),
            "step": int(step)
        }
        with open(filename, "wb") as f:
            pickle.dump(flat_data, f)

    def trigger_rollback(self, filename="spark_3_rollback.pkl"):
        if not os.path.exists(filename):
            return None, None
        with open(filename, "rb") as f:
            data = pickle.load(f)
            
        restored_params = jax.device_put(data["params"])
        restored_opt_state = jax.device_put(data["opt_state"])
        
        print(f"🔄 [AUDITOR] Rollback executed. Restoring parameters from step {data['step']}.")
        return restored_params, restored_opt_state
    
    def update_probe_scores(self, summary):
        for probe, score in summary.items():
            if probe not in self.probe_scores:
                self.probe_scores[probe] = []
            self.probe_scores[probe].append(float(score))
        self.probe_latest = summary
        self.probe_count += 1

    def diagnose_probes(self, threshold=50.0):
        if self.probe_count == 0:
            return "No probe data collected yet."
        msgs = []
        for probe, scores in self.probe_scores.items():
            if len(scores) < 2:
                continue
            recent = scores[-3:]
            avg = sum(recent) / len(recent)
            if avg < threshold:
                msgs.append(f"⚠️  {probe}: {avg:.1f}% — consider adding examples.")
            elif avg > 90.0:
                msgs.append(f"✅ {probe}: {avg:.1f}% — mastered.")
            else:
                msgs.append(f"📈 {probe}: {avg:.1f}% — progressing.")
        return "\n".join(msgs) if msgs else "Probe scores look healthy."

    def get_low_scores(self, threshold=50.0):
        low = []
        for probe, scores in self.probe_scores.items():
            if len(scores) >= 2 and (sum(scores[-3:]) / len(scores[-3:])) < threshold:
                low.append(probe)
        return low