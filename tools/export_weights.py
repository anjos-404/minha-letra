"""Converte o checkpoint TF1 do modelo pré-treinado em um .npz (NumPy).

Uso: python -I export_weights.py <prefixo_do_checkpoint> <saida.npz>
Só lê os tensores (index + data); não executa nenhum código do checkpoint.
"""
import sys

import numpy as np
import tensorflow as tf

ckpt, out = sys.argv[1], sys.argv[2]
reader = tf.train.load_checkpoint(ckpt)
weights = {}
for name, shape in sorted(reader.get_variable_to_shape_map().items()):
    low = name.lower()
    if any(s in low for s in ("rmsprop", "adam", "global_step", "ema", "power")):
        continue
    print(name, shape)
    weights[name.replace("/", "__")] = reader.get_tensor(name).astype(np.float32)
np.savez_compressed(out, **weights)
print("salvo em", out)
