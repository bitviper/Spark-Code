# spark_data_f.py
# Data pipeline: BPE tokenizer training, data loading, character‑level tokenizer,
# and helper to extend a causal checkpoint into a BERT model.
#
# Critical fixes:
# - No shuffling of flat token arrays (preserves syntax)
# - Forced [MASK] token at end of vocabulary
# - CharMaskTokenizer includes punctuation
# - extend_base_model adds pos_embeddings for BERT conversion

import os
import numpy as np
import jax.numpy as jnp
import jax
import sentencepiece as spm

def build_tokenizer(corpus_path, model_prefix="spark_mask", vocab_size=4000,
                    user_defined_symbols=None):
    if not os.path.exists(model_prefix + ".model"):
        print(f"Training BPE tokenizer on {corpus_path}...")
        
        symbols = user_defined_symbols or []
        if "🔲" not in symbols:
            symbols.append("🔲")

        spm.SentencePieceTrainer.train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            user_defined_symbols=symbols,
            model_type='bpe',
            character_coverage=1.0,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            mask_id=vocab_size - 1 
        )
        print("Tokenizer trained.")
        
    sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    return sp

def load_training_data(sp, domain_path, original_path=None, ufo_path=None,
                       mix_ratio=0.5, total_needed=2_000_000, max_chars=5_000_000):
    domain_text = ""
    if os.path.exists(domain_path):
        with open(domain_path, "r", encoding="utf-8") as f:
            domain_text += f.read(max_chars)
    if ufo_path and os.path.exists(ufo_path):
        with open(ufo_path, "r", encoding="utf-8") as f:
            domain_text += f.read(max_chars)

    original_text = ""
    if original_path and os.path.exists(original_path):
        with open(original_path, "r", encoding="utf-8") as f:
            original_text = f.read(max_chars)

    domain_ids = sp.encode(domain_text) if domain_text else []
    original_ids = sp.encode(original_text) if original_text else []

    domain_len = min(len(domain_ids), int(total_needed * (1 - mix_ratio)))
    original_len = min(len(original_ids), int(total_needed * mix_ratio))

    domain_ids = domain_ids[:domain_len]
    original_ids = original_ids[:original_len]

    all_ids = np.concatenate([domain_ids, original_ids])
    # No shuffle – training loops randomly slice sub‑sequences
    print(f"Tokens: domain={len(domain_ids)}, original={len(original_ids)}, total={len(all_ids)}")
    return jnp.array(all_ids, dtype=jnp.int32)

def split_train_val(ids, val_frac=0.05):
    split = int(len(ids) * (1 - val_frac))
    return ids[:split], ids[split:]

class CharMaskTokenizer:
    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyz .,!?;:'-"
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.mask_token = "[M]"
        self.mask_id = len(chars)
        self.char_to_id[self.mask_token] = self.mask_id
        self.id_to_char[self.mask_id] = self.mask_token
        self.vocab_size = len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id.get(c, self.char_to_id[' ']) for c in text.lower()]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids if i != self.mask_id)

def extend_base_model(base_ckpt, new_seq_len, d_model):
    import pickle
    with open(base_ckpt, "rb") as f:
        data = pickle.load(f)
        base_params = data["params"]
        old_config = data.get("config", {})

    old_emb = base_params["tok_embeddings"]["weight"]
    old_vocab = old_emb.shape[0]
    
    key = jax.random.PRNGKey(42)
    new_row = jax.random.normal(key, (1, d_model)) * 0.02
    new_emb = jnp.concatenate([old_emb, new_row], axis=0)

    params = base_params.copy()
    params["tok_embeddings"] = {"weight": new_emb}
    
    pos_key = jax.random.PRNGKey(43)
    params["pos_embeddings"] = {
        "weight": jax.random.normal(pos_key, (new_seq_len, d_model)) * 0.02
    }

    config = old_config.copy()
    config["vocab_size"] = old_vocab + 1
    config["mask_token_id"] = old_vocab
    config["seq_len"] = new_seq_len
    config.setdefault("dropout_rate", 0.1)
    config.setdefault("lr", 5e-5)
    config.setdefault("batch_size", 4)
    config.setdefault("chunk_size", 20)
    
    print(f"Extended model: Vocab {old_vocab} -> {config['vocab_size']}. Added pos_embeddings for seq_len {new_seq_len}.")
    return params, config