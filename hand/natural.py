"""Variações humanas: o que faz a escrita parecer feita à mão, e não impressa.

Duas camadas, ambas controladas por "naturalidade" (0 = certinha, 1 = normal,
2 = bem solta):

1. Por linha (humanize): ninguém escreve perfeitamente reto.
   - a linha de base ondula de leve ao longo da linha;
   - a inclinação das letras muda devagar (uma palavra mais deitada, outra mais em pé);
   - cada linha sai num ângulo, num tamanho e com um começo um pouco diferentes.

2. Por traço (stroke_svg): pressão da caneta.
   - o modelo gera um ponto por instante de tempo, então a distância entre pontos
     é a velocidade da mão: onde ela corre, o traço afina; onde desacelera, engrossa;
   - o traço afina no começo e no fim (a caneta encosta e levanta);
   - onde a pressão é menor, a tinta fica um pouco mais clara.
"""
from __future__ import annotations

import numpy as np


def _smooth_noise(n: int, rng, width: int = 9) -> np.ndarray:
    if n <= 0:
        return np.zeros(0)
    k = np.ones(width) / width
    raw = rng.normal(0, 1, n + width)
    sm = np.convolve(raw, k, mode="same")[width // 2: width // 2 + n]
    return sm / max(np.std(sm), 1e-6) if n > 3 else sm


def humanize(strokes: list, *, x0: float, base: float, xh: float, amount: float, rng,
             max_x: float | None = None, line_level: bool = True) -> list:
    """Aplica as variações de uma linha (px, y para baixo, base = y da pauta)."""
    if amount <= 0 or not strokes:
        return strokes
    a = float(amount)
    # parâmetros desta linha
    tilt = np.radians(rng.normal(0, 0.45) * a) if line_level else 0.0           # linha um pouco torta
    slant0 = np.radians(rng.normal(0, 1.6) * a)                                  # inclinação média
    slant_amp = np.radians(1.4 * a)                                              # inclinação variando
    slant_len, slant_ph = rng.uniform(220, 380), rng.uniform(0, 2 * np.pi)
    wob_amp = 0.06 * xh * a                                                      # linha de base ondulando
    wob_len, wob_ph = rng.uniform(170, 320), rng.uniform(0, 2 * np.pi)
    wob2_len, wob2_ph = rng.uniform(60, 110), rng.uniform(0, 2 * np.pi)
    scale = (1.0 + rng.normal(0, 0.025) * a) if line_level else 1.0             # tamanho da linha
    dx = (rng.uniform(-0.15, 0.55) * xh * a) if line_level else 0.0             # começo da linha
    dy = (rng.normal(0, 0.05) * xh * a) if line_level else 0.0

    out = []
    for s in strokes:
        s = np.asarray(s, float)
        if s.ndim != 2 or len(s) == 0:
            out.append(s)
            continue
        x = s[:, 0] - x0
        y = s[:, 1] - base
        # tamanho e ângulo da linha
        x, y = x * scale, y * scale
        c, sn = np.cos(tilt), np.sin(tilt)
        x, y = c * x - sn * y, sn * x + c * y
        # inclinação variando ao longo da linha (cisalhamento em torno da base)
        slant = slant0 + slant_amp * np.sin(2 * np.pi * x / slant_len + slant_ph)
        x = x - y * np.tan(slant)
        # base ondulando (uma onda longa + uma curta e mais fraca)
        y = y + wob_amp * (np.sin(2 * np.pi * x / wob_len + wob_ph) + 0.35 * np.sin(2 * np.pi * x / wob2_len + wob2_ph))
        out.append(np.stack([x + x0 + dx, y + base + dy], 1))
    if max_x is not None:
        right = max(float(s[:, 0].max()) for s in out if len(s))
        if right > max_x:
            out = [s - [right - max_x, 0.0] for s in out]
    return out


def _pressure(s: np.ndarray, ref_step: float, rng) -> np.ndarray:
    """Pressão (≈1 em média) em cada ponto do traço."""
    n = len(s)
    if n < 2:
        return np.ones(n)
    seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
    v = np.concatenate([[seg[0]], (seg[:-1] + seg[1:]) / 2, [seg[-1]]]) / max(ref_step, 1e-6)
    p = 1.0 - 0.42 * np.clip(v - 1.0, -0.9, 1.6)          # mais rápido = mais leve
    k = min(5, n)
    p = np.convolve(np.pad(p, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")[:n]
    p = p + 0.12 * _smooth_noise(n, rng, 11)              # a mão nunca aperta igual
    # caneta encostando e levantando
    L = np.concatenate([[0], np.cumsum(seg)])
    total = L[-1]
    ramp = max(min(total * 0.12, 9.0), 1e-6)
    taper = np.clip(np.minimum(L, total - L) / ramp, 0, 1)
    p = p * (0.55 + 0.45 * np.sqrt(taper))
    return np.clip(p, 0.45, 1.45)


def stroke_svg(strokes: list, *, base_w: float, amount: float, rng, group_class: str = "st") -> str:
    """Traços -> SVG. Com naturalidade > 0, cada traço vira pedacinhos com
    espessura e tinta variando (pressão). Cada traço fica num <g> próprio, para a
    animação de escrita saber onde a caneta levanta."""
    strokes = [np.asarray(s, float) for s in strokes if s is not None and len(s)]
    if not strokes:
        return ""
    if amount <= 0:
        return "".join(f'<path d="{_d(s)}"/>' for s in strokes)
    steps = np.concatenate([np.linalg.norm(np.diff(s, axis=0), axis=1) for s in strokes if len(s) > 1] or [np.ones(1)])
    ref = float(np.median(steps)) if len(steps) else 1.0
    amt = min(float(amount), 1.6)
    parts = []
    for s in strokes:
        if len(s) < 3:
            parts.append(f'<g class="{group_class}"><path d="{_d(s)}" stroke-width="{base_w * 0.9:.2f}"/></g>')
            continue
        p = _pressure(s, ref, rng)
        w = base_w * (1.0 + amt * 0.55 * (p - 1.0))
        op = np.clip(1.0 - amt * 0.28 * (1.12 - p), 0.62, 1.0)
        wq = np.round(w * 8) / 8           # agrupa pedaços de espessura parecida
        oq = np.round(op * 20) / 20
        segs, start = [], 0
        for i in range(1, len(s)):
            change = (wq[i] != wq[start] or oq[i] != oq[start]) and i - start >= 2
            if change or i == len(s) - 1:
                end = i
                segs.append((start, end, float(w[start:end + 1].mean()), float(op[start:end + 1].mean())))
                start = i
        body = "".join(
            f'<path d="{_d(s[a:b + 1])}" stroke-width="{ww:.2f}"' + (f' stroke-opacity="{oo:.2f}"' if oo < 0.99 else "") + "/>"
            for a, b, ww, oo in segs)
        parts.append(f'<g class="{group_class}">{body}</g>')
    return "".join(parts)


def _d(s: np.ndarray) -> str:
    if len(s) == 1:
        s = np.vstack([s, s + [0.6, 0.6]])
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in s[:, :2])
