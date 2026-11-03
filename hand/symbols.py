"""Sinais que o modelo não aprendeu (+ = / × ÷ % < > ...), desenhados como
traços de caneta.

Cada sinal é descrito em unidades da altura do "x" da letra da pessoa
(xh = 1), com y para baixo, a linha de base em y = 0 e o eixo matemático
(onde ficam o traço do +, do = e da fração) em y = -0.5.
"""
from __future__ import annotations

import numpy as np

AXIS = -0.5


def _seg(a, b, n: int = 10) -> np.ndarray:
    t = np.linspace(0, 1, n)[:, None]
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def _arc(cx, cy, rx, ry, a0, a1, n: int = 18) -> np.ndarray:
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)


def _dot(cx, cy, r: float = 0.07) -> np.ndarray:
    return _arc(cx, cy, r, r, 0, 360, 9)


def _smooth(p: np.ndarray, rounds: int = 3) -> np.ndarray:
    """Suaviza uma linha poligonal (cortes de Chaikin)."""
    for _ in range(rounds):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], np.column_stack([q, r]).reshape(-1, 2), p[-1:]])
    return p


def _poly(*pts, n: int = 8) -> np.ndarray:
    return np.vstack([_seg(pts[i], pts[i + 1], n)[:-1] if i < len(pts) - 2 else _seg(pts[i], pts[i + 1], n)
                      for i in range(len(pts) - 1)])


def shape(ch: str):
    """(lista de traços, largura) do sinal, em unidades de xh. None se não sabe."""
    a = AXIS
    if ch == "+":
        return [_seg((0.1, a), (0.8, a)), _seg((0.45, a - 0.36), (0.45, a + 0.36))], 0.9
    if ch == "−":
        return [_seg((0.12, a), (0.78, a))], 0.9
    if ch == "=":
        return [_seg((0.1, a - 0.17), (0.8, a - 0.17)), _seg((0.1, a + 0.17), (0.8, a + 0.17))], 0.9
    if ch == "≠":
        s, w = shape("=")
        return s + [_seg((0.62, a - 0.45), (0.28, a + 0.45))], w
    if ch in "×*":
        return [_seg((0.15, a - 0.3), (0.65, a + 0.3)), _seg((0.65, a - 0.3), (0.15, a + 0.3))], 0.8
    if ch == "÷":
        return [_seg((0.1, a), (0.8, a)), _dot(0.45, a - 0.3), _dot(0.45, a + 0.3)], 0.9
    if ch == "·":
        return [_dot(0.25, a, 0.06)], 0.5
    if ch == "/":
        return [_seg((0.05, 0.25), (0.6, -1.35))], 0.65
    if ch == "\\":
        return [_seg((0.05, -1.35), (0.6, 0.25))], 0.65
    if ch == "<":
        return [_poly((0.75, a - 0.35), (0.12, a), (0.75, a + 0.35))], 0.85
    if ch == ">":
        return [_poly((0.12, a - 0.35), (0.75, a), (0.12, a + 0.35))], 0.85
    if ch == "≤":
        return [_poly((0.75, a - 0.45), (0.12, a - 0.12), (0.75, a + 0.2)), _seg((0.12, a + 0.42), (0.75, a + 0.42))], 0.85
    if ch == "≥":
        return [_poly((0.12, a - 0.45), (0.75, a - 0.12), (0.12, a + 0.2)), _seg((0.12, a + 0.42), (0.75, a + 0.42))], 0.85
    if ch == "%":
        return [_arc(0.22, -1.0, 0.16, 0.2, 0, 360), _seg((0.05, 0.1), (0.85, -1.3)), _arc(0.7, -0.25, 0.16, 0.2, 0, 360)], 0.95
    if ch == "°":
        return [_arc(0.2, -1.15, 0.14, 0.15, 0, 360)], 0.45
    if ch == "(":
        return [_S((0.42, -1.42), (0.2, -1.1), (0.1, -0.55), (0.18, 0.05), (0.42, 0.38))], 0.5
    if ch == ")":
        return [_S((0.08, -1.42), (0.3, -1.1), (0.4, -0.55), (0.32, 0.05), (0.08, 0.38))], 0.5
    if ch == "[":
        return [_poly((0.45, -1.4), (0.12, -1.4), (0.12, 0.35), (0.45, 0.35))], 0.55
    if ch == "]":
        return [_poly((0.1, -1.4), (0.43, -1.4), (0.43, 0.35), (0.1, 0.35))], 0.55
    if ch == "{":
        up = _arc(0.5, -1.4 + 0.25, 0.25, 0.25, 270, 180, 6)
        top = _seg((0.25, -1.15), (0.25, -0.75), 5)
        mid = np.vstack([_arc(0.05, -0.75, 0.2, 0.2, 0, 90, 5), _arc(0.05, -0.35, 0.2, 0.2, 270, 360, 5)])
        bot = _seg((0.25, -0.35), (0.25, 0.1), 5)
        down = _arc(0.5, 0.1, 0.25, 0.25, 180, 90, 6)
        return [np.vstack([up, top, mid, bot, down])], 0.65
    if ch == "}":
        s, w = shape("{")
        return [np.stack([0.6 - s[0][:, 0], s[0][:, 1]], 1)], w
    if ch == "|":
        return [_seg((0.25, -1.4), (0.25, 0.35))], 0.5
    if ch == "→":
        return [_seg((0.05, a), (1.05, a)), _poly((0.8, a - 0.22), (1.05, a), (0.8, a + 0.22))], 1.15
    if ch == "←":
        return [_seg((0.1, a), (1.1, a)), _poly((0.35, a - 0.22), (0.1, a), (0.35, a + 0.22))], 1.15
    if ch == "~":
        t = np.linspace(0, 1, 14)
        return [np.stack([0.08 + 0.7 * t, a - 0.12 * np.sin(t * 2 * np.pi)], 1)], 0.85
    if ch == "_":
        return [_seg((0.0, 0.18), (0.8, 0.18))], 0.8
    if ch == "√":
        return [_poly((0.0, a + 0.05), (0.15, a - 0.05), (0.38, 0.15), (0.75, -1.45))], 0.8
    if ch == "±":
        return [_seg((0.1, a - 0.15), (0.8, a - 0.15)), _seg((0.45, a - 0.5), (0.45, a + 0.2)),
                _seg((0.1, a + 0.4), (0.8, a + 0.4))], 0.9
    if ch == "∞":
        t = np.linspace(0, 2 * np.pi, 30)
        return [np.stack([0.5 + 0.42 * np.sin(t), a + 0.2 * np.sin(2 * t)], 1)], 1.0
    if ch == "π":
        return [_seg((0.05, -0.9), (0.85, -0.95)), _seg((0.28, -0.92), (0.22, 0.0)), _poly((0.62, -0.93), (0.64, -0.1), (0.75, 0.0))], 0.9
    if ch == "≈":
        t = np.linspace(0, 1, 14)
        w1 = np.stack([0.08 + 0.7 * t, a - 0.17 - 0.1 * np.sin(t * 2 * np.pi)], 1)
        w2 = np.stack([0.08 + 0.7 * t, a + 0.17 - 0.1 * np.sin(t * 2 * np.pi)], 1)
        return [w1, w2], 0.85
    if ch == "@":
        outer = _arc(0.55, a, 0.5, 0.55, 20, 340, 24)
        inner = _arc(0.5, a, 0.18, 0.22, 0, 360, 12)
        return [np.vstack([inner, _seg((0.68, a - 0.2), (0.7, a + 0.2), 3), outer])], 1.1
    if ch == "&":
        return [np.vstack([_seg((0.8, 0.0), (0.2, -0.95), 8), _arc(0.35, -1.0, 0.18, 0.22, 330, 180, 8),
                           _seg((0.18, -1.0), (0.45, -0.45), 5), _arc(0.3, -0.25, 0.25, 0.25, 300, 90, 10),
                           _seg((0.3, 0.0), (0.75, -0.45), 5)])], 0.9
    if ch == "$":
        s = _smooth(np.array([(0.75, -1.0), (0.48, -1.18), (0.18, -1.02), (0.22, -0.72), (0.5, -0.58),
                              (0.76, -0.4), (0.74, -0.1), (0.45, 0.02), (0.12, -0.12)]))
        return [s, _seg((0.45, -1.4), (0.45, 0.3))], 0.9
    if ch == ".":
        return [_dot(0.12, -0.03, 0.05)], 0.3
    if ch == ":":
        return [_dot(0.12, -0.55, 0.05), _dot(0.12, -0.03, 0.05)], 0.3
    if ch == ";":
        return [_dot(0.12, -0.55, 0.05), _poly((0.12, -0.02), (0.13, 0.12), (0.05, 0.3), n=5)], 0.3
    if ch == ",":
        return [_poly((0.12, -0.02), (0.13, 0.12), (0.05, 0.3), n=5)], 0.25
    if ch in "✓✔":
        return [_poly((0.05, -0.5), (0.35, -0.05), (0.98, -1.25), n=8)], 1.05
    if ch in "✗✘":
        return [_seg((0.1, -1.0), (0.8, 0.0)), _seg((0.8, -1.0), (0.1, 0.0))], 0.9
    if ch in "•●":
        return [_dot(0.3, -0.5, 0.12), _dot(0.3, -0.5, 0.06)], 0.6
    if ch == "€":
        return [_arc(0.6, -0.6, 0.5, 0.65, 50, 310, 20), _seg((0.0, -0.75), (0.65, -0.75)), _seg((0.0, -0.45), (0.6, -0.45))], 1.0
    if ch == "#":
        return [_seg((0.3, -1.1), (0.2, 0.1)), _seg((0.65, -1.1), (0.55, 0.1)),
                _seg((0.05, -0.75), (0.85, -0.75)), _seg((0.0, -0.3), (0.8, -0.3))], 0.9
    if ch in GREEK_SHAPES:
        strokes, w = GREEK_SHAPES[ch]
        return [s.copy() for s in strokes], w
    if ch.startswith("comp:"):
        return component(ch[5:])
    return None


def _S(*pts) -> np.ndarray:
    """Curva suave passando perto dos pontos dados."""
    return _smooth(np.array(pts, float))


def _oval(cx, cy, rx, ry) -> np.ndarray:
    return _arc(cx, cy, rx, ry, -100, 260, 26)


# Letras que o modelo não aprendeu: gregas, operadores de cálculo e as
# maiúsculas Q, X, Z (faltavam nos dados de treino).
# Minúsculas vão de 0 a -1 (altura do x); maiúsculas e hastes até -1.4;
# perninhas descem até +0.45.
GREEK_SHAPES = {
    "α": ([_S((0.8, -0.95), (0.62, -0.4), (0.42, -0.04), (0.16, -0.08), (0.08, -0.48), (0.26, -0.9),
             (0.52, -0.88), (0.66, -0.35), (0.82, 0.0))], 0.9),
    "β": ([_S((0.12, 0.45), (0.12, -0.6), (0.14, -1.1), (0.32, -1.42), (0.6, -1.3), (0.56, -0.86), (0.3, -0.76),
             (0.7, -0.55), (0.66, -0.1), (0.36, 0.02), (0.13, -0.12))], 0.85),
    "γ": ([_S((0.04, -0.92), (0.24, -0.88), (0.44, -0.2), (0.4, 0.34), (0.3, 0.42), (0.3, 0.2), (0.5, -0.35),
             (0.76, -0.95))], 0.85),
    "δ": ([_S((0.66, -1.32), (0.42, -1.45), (0.24, -1.22), (0.6, -0.88), (0.72, -0.42), (0.46, 0.02), (0.14, -0.2),
             (0.2, -0.7), (0.5, -0.86))], 0.8),
    "ε": ([_S((0.66, -0.88), (0.36, -1.0), (0.14, -0.8), (0.36, -0.55), (0.14, -0.35), (0.24, -0.02),
             (0.7, -0.1))], 0.75),
    "η": ([_S((0.04, -0.82), (0.14, -0.96), (0.2, -0.7)), _seg((0.2, -0.7), (0.2, 0.0)),
           _S((0.2, -0.55), (0.42, -0.95), (0.66, -0.78), (0.68, -0.3), (0.68, 0.45))], 0.85),
    "θ": ([_oval(0.38, -0.72, 0.3, 0.72), _seg((0.1, -0.72), (0.66, -0.72))], 0.8),
    "λ": ([_seg((0.16, -1.45), (0.3, -1.32), 4), _seg((0.3, -1.32), (0.76, 0.0)), _seg((0.47, -0.78), (0.08, 0.0))], 0.85),
    "μ": ([_seg((0.12, 0.45), (0.12, -0.95)),
           _S((0.12, -0.32), (0.3, 0.0), (0.55, -0.2), (0.62, -0.95)), _S((0.62, -0.95), (0.62, -0.12), (0.8, 0.0))], 0.9),
    "ν": ([_S((0.04, -0.92), (0.18, -0.95), (0.36, 0.0), (0.62, -0.5), (0.7, -0.95))], 0.8),
    "ρ": ([_S((0.12, 0.45), (0.14, -0.4), (0.34, -0.96), (0.66, -0.76), (0.66, -0.24), (0.4, 0.0), (0.15, -0.2))], 0.8),
    "σ": ([_S((0.88, -0.95), (0.42, -0.95), (0.12, -0.6), (0.2, -0.06), (0.5, 0.0), (0.7, -0.36), (0.6, -0.86),
             (0.4, -0.95))], 0.9),
    "τ": ([_S((0.04, -0.82), (0.24, -0.95), (0.82, -0.95)), _S((0.46, -0.95), (0.42, -0.2), (0.55, 0.0), (0.7, -0.1))], 0.85),
    "φ": ([_oval(0.42, -0.5, 0.33, 0.46), _seg((0.42, -1.4), (0.42, 0.45))], 0.85),
    "ϕ": ([_oval(0.42, -0.5, 0.33, 0.46), _seg((0.42, -1.4), (0.42, 0.45))], 0.85),
    "χ": ([_S((0.04, -0.9), (0.18, -0.95), (0.62, 0.35), (0.76, 0.42)), _seg((0.7, -0.95), (0.1, 0.42))], 0.8),
    "ψ": ([_S((0.06, -0.92), (0.12, -0.3), (0.42, -0.06), (0.72, -0.3), (0.8, -0.92)), _seg((0.44, -1.4), (0.42, 0.45))], 0.9),
    "ω": ([_S((0.16, -0.92), (0.05, -0.4), (0.2, 0.0), (0.42, -0.28), (0.46, -0.56), (0.5, -0.28), (0.72, 0.0),
             (0.88, -0.4), (0.78, -0.92))], 0.95),
    "Δ": ([_poly((0.06, 0.0), (0.5, -1.4), (0.94, 0.0), (0.06, 0.0))], 1.0),
    "Ω": ([np.vstack([_seg((0.08, 0.0), (0.36, 0.0), 4), _S((0.36, 0.0), (0.36, -0.14), (0.14, -0.5), (0.12, -0.98),
                                                           (0.5, -1.4), (0.88, -0.98), (0.86, -0.5), (0.64, -0.14),
                                                           (0.64, 0.0)), _seg((0.64, 0.0), (0.92, 0.0), 4)])], 1.0),
    "Σ": ([_poly((0.86, -1.4), (0.1, -1.4), (0.56, -0.7), (0.1, 0.0), (0.88, 0.0))], 0.95),
    "Φ": ([_oval(0.5, -0.7, 0.42, 0.42), _seg((0.5, -1.45), (0.5, 0.05))], 1.0),
    "Θ": ([_oval(0.5, -0.7, 0.4, 0.7), _seg((0.18, -0.7), (0.82, -0.7))], 1.0),
    "Π": ([_seg((0.06, -1.4), (0.94, -1.4)), _seg((0.26, -1.4), (0.26, 0.0)), _seg((0.74, -1.4), (0.74, 0.0))], 1.0),
    "Λ": ([_poly((0.05, 0.0), (0.5, -1.4), (0.95, 0.0))], 1.0),
    "Γ": ([_poly((0.16, 0.0), (0.16, -1.4), (0.86, -1.4))], 0.9),
    "Ψ": ([_S((0.06, -1.3), (0.12, -0.75), (0.5, -0.45), (0.88, -0.75), (0.94, -1.3)), _seg((0.5, -1.4), (0.5, 0.0))], 1.0),
    "∂": ([_S((0.14, -1.24), (0.44, -1.45), (0.7, -1.1), (0.7, -0.4), (0.46, 0.02), (0.15, -0.16), (0.14, -0.6),
             (0.44, -0.76), (0.7, -0.56))], 0.85),
    "∇": ([_poly((0.05, -1.4), (0.95, -1.4), (0.5, 0.0), (0.05, -1.4))], 1.0),
    "∠": ([_poly((0.86, -1.1), (0.1, 0.0), (0.92, 0.0))], 1.0),
    "⊥": ([_seg((0.5, -1.3), (0.5, 0.0)), _seg((0.08, 0.0), (0.92, 0.0))], 1.0),
    "∥": ([_seg((0.25, -1.35), (0.25, 0.3)), _seg((0.55, -1.35), (0.55, 0.3))], 0.8),
    "′": ([_seg((0.22, -1.4), (0.12, -0.95), 5)], 0.3),
    "∈": ([_S((0.85, -1.0), (0.4, -1.0), (0.12, -0.5), (0.4, 0.0), (0.85, 0.0)), _seg((0.15, -0.5), (0.72, -0.5))], 0.95),
    "∴": ([_dot(0.45, -0.85, 0.06), _dot(0.12, -0.05, 0.06), _dot(0.78, -0.05, 0.06)], 0.9),
    "⇒": ([_seg((0.05, -0.68), (0.85, -0.68)), _seg((0.05, -0.32), (0.85, -0.32)),
           _poly((0.72, -0.9), (1.08, -0.5), (0.72, -0.1))], 1.15),
    "⇔": ([_seg((0.25, -0.68), (0.95, -0.68)), _seg((0.25, -0.32), (0.95, -0.32)),
           _poly((0.35, -0.9), (0.02, -0.5), (0.35, -0.1)), _poly((0.85, -0.9), (1.18, -0.5), (0.85, -0.1))], 1.2),
    "↑": ([_seg((0.4, 0.0), (0.4, -1.3)), _poly((0.15, -1.0), (0.4, -1.3), (0.65, -1.0))], 0.8),
    "↓": ([_seg((0.4, -1.3), (0.4, 0.0)), _poly((0.15, -0.3), (0.4, 0.0), (0.65, -0.3))], 0.8),
    "⋯": ([_dot(0.15, -0.5, 0.06), _dot(0.5, -0.5, 0.06), _dot(0.85, -0.5, 0.06)], 1.0),
    "⋮": ([_dot(0.25, -1.1, 0.06), _dot(0.25, -0.6, 0.06), _dot(0.25, -0.1, 0.06)], 0.5),
    "⋱": ([_dot(0.1, -1.1, 0.06), _dot(0.45, -0.6, 0.06), _dot(0.8, -0.1, 0.06)], 0.9),
    "j": ([_S((0.42, -0.95), (0.42, -0.2), (0.4, 0.25), (0.25, 0.42), (0.06, 0.32)), _dot(0.43, -1.3, 0.06)], 0.55),
    "I": ([_seg((0.42, -1.4), (0.4, 0.0)), _seg((0.12, -1.4), (0.72, -1.42)), _seg((0.1, 0.0), (0.7, -0.02))], 0.82),
    "t": ([_S((0.36, -1.42), (0.34, -0.7), (0.35, -0.12), (0.45, 0.0), (0.62, -0.06), (0.7, -0.16)),
           _seg((0.08, -0.93), (0.66, -0.97))], 0.78),
    "Q": ([_oval(0.52, -0.7, 0.44, 0.7), _seg((0.62, -0.28), (0.98, 0.1))], 1.1),
    "X": ([_seg((0.08, -1.4), (0.9, 0.0)), _seg((0.9, -1.4), (0.08, 0.0))], 1.0),
    "Z": ([_poly((0.1, -1.4), (0.88, -1.4), (0.1, 0.0), (0.92, 0.0))], 1.0),
    # operadores grandes (desenhados mais altos; a montagem os posiciona)
    "∫": ([_S((0.66, -1.95), (0.52, -2.08), (0.4, -1.9), (0.34, -0.6), (0.3, 0.6), (0.18, 0.84), (0.04, 0.72))], 0.7),
    "∮": ([_S((0.66, -1.95), (0.52, -2.08), (0.4, -1.9), (0.34, -0.6), (0.3, 0.6), (0.18, 0.84), (0.04, 0.72)),
           _oval(0.34, -0.62, 0.22, 0.26)], 0.7),
    "∑": ([_poly((1.3, -1.75), (0.1, -1.75), (0.8, -0.5), (0.1, 0.75), (1.35, 0.75))], 1.45),
    "∏": ([_seg((0.05, -1.75), (1.35, -1.75)), _seg((0.32, -1.75), (0.32, 0.75)), _seg((1.08, -1.75), (1.08, 0.75))], 1.4),
}
GREEK_SHAPES["∆"] = GREEK_SHAPES["Δ"]
GREEK_SHAPES["Ω"] = GREEK_SHAPES["Ω"]
GREEK_SHAPES["µ"] = GREEK_SHAPES["μ"]          # micro (teclado) = mu
GREEK_SHAPES["ϵ"] = GREEK_SHAPES["ε"]
GREEK_SHAPES["ϑ"] = GREEK_SHAPES["θ"]
GREEK_SHAPES["∑"] = GREEK_SHAPES["∑"]
GREEK_SHAPES["Σ"] = GREEK_SHAPES["Σ"]

LETTERLIKE = set("jtIαβγδεηθλμνρστφϕχψωΔΩΣΦΘΠΛΓΨ∂∇QXZ∆Ωµϵϑ′")
BIG_OPS = set("∫∮∑∏")


# ---------------- símbolos de circuito ----------------
COMPONENT_NAMES = {
    "resistor": "resistor", "resistência": "resistor", "resistencia": "resistor",
    "capacitor": "capacitor", "condensador": "capacitor",
    "indutor": "indutor", "bobina": "indutor",
    "bateria": "bateria", "pilha": "bateria", "fonte dc": "bateria", "fonte cc": "bateria",
    "fonte": "fonte ac", "fonte ac": "fonte ac", "fonte ca": "fonte ac", "gerador": "fonte ac",
    "terra": "terra", "aterramento": "terra",
    "diodo": "diodo", "led": "led",
    "lâmpada": "lâmpada", "lampada": "lâmpada",
    "chave": "chave", "interruptor": "chave",
    "amperímetro": "amperímetro", "amperimetro": "amperímetro",
    "voltímetro": "voltímetro", "voltimetro": "voltímetro",
    "fio": "fio",
    "fonte de corrente": "fonte corrente", "fonte corrente": "fonte corrente",
    "fusível": "fusível", "fusivel": "fusível", "motor": "motor",
    "ohmímetro": "ohmímetro", "ohmimetro": "ohmímetro",
    "potenciômetro": "potenciômetro", "potenciometro": "potenciômetro", "pot": "potenciômetro",
    "chave fechada": "chave fechada", "interruptor fechado": "chave fechada",
    "chave aberta": "chave", "interruptor aberto": "chave", "impedância": "impedância", "impedancia": "impedância",
}


def component(name: str):
    """Componentes de circuito, alinhados no eixo (y = -0.5), com terminais nas pontas."""
    a = AXIS
    if name == "resistor":
        xs = np.linspace(0.45, 2.15, 9)
        ys = [a] + [a - 0.32 if i % 2 else a + 0.32 for i in range(1, 8)] + [a]
        return [_seg((0.0, a), (0.45, a), 4), _poly(*zip(xs, ys), n=4), _seg((2.15, a), (2.6, a), 4)], 2.6
    if name == "capacitor":
        return [_seg((0.0, a), (1.1, a)), _seg((1.1, a - 0.55), (1.1, a + 0.55)),
                _seg((1.42, a - 0.55), (1.42, a + 0.55)), _seg((1.42, a), (2.52, a))], 2.52
    if name == "indutor":
        bumps = [_arc(0.66 + 0.42 * k, a, 0.21, 0.32, 180, 360, 9) for k in range(4)]
        return [_seg((0.0, a), (0.45, a), 4), np.vstack(bumps), _seg((2.07, a), (2.52, a), 4)], 2.52
    if name == "bateria":
        return [_seg((0.0, a), (1.1, a)), _seg((1.1, a - 0.62), (1.1, a + 0.62)),
                _seg((1.4, a - 0.3), (1.4, a + 0.3)), _seg((1.4, a), (2.5, a)),
                _seg((0.7, a - 0.72), (0.98, a - 0.72), 4), _seg((0.84, a - 0.86), (0.84, a - 0.58), 4)], 2.5
    if name == "fonte ac":
        t = np.linspace(0, 1, 16)
        wave = np.stack([0.95 + 0.6 * t, a - 0.2 * np.sin(t * 2 * np.pi)], 1)
        return [_seg((0.0, a), (0.62, a)), _oval(1.25, a, 0.63, 0.63), wave, _seg((1.88, a), (2.5, a))], 2.5
    if name == "terra":
        return [_seg((0.6, a - 0.9), (0.6, a + 0.1)), _seg((0.1, a + 0.1), (1.1, a + 0.1)),
                _seg((0.27, a + 0.32), (0.93, a + 0.32)), _seg((0.45, a + 0.54), (0.75, a + 0.54))], 1.2
    if name in ("diodo", "led"):
        parts = [_seg((0.0, a), (0.8, a)), _poly((0.8, a - 0.45), (0.8, a + 0.45), (1.55, a), (0.8, a - 0.45)),
                 _seg((1.55, a - 0.45), (1.55, a + 0.45)), _seg((1.55, a), (2.35, a))]
        if name == "led":
            for dx in (0.0, 0.32):
                parts += [_seg((1.2 + dx, a - 0.55), (1.5 + dx, a - 0.95), 5),
                          _poly((1.32 + dx, a - 0.92), (1.5 + dx, a - 0.95), (1.5 + dx, a - 0.76), n=3)]
        return parts, 2.35
    if name == "lâmpada":
        r = 0.6
        d = r * 0.7
        return [_seg((0.0, a), (0.65, a)), _oval(1.25, a, r, r), _seg((1.25 - d, a - d), (1.25 + d, a + d)),
                _seg((1.25 + d, a - d), (1.25 - d, a + d)), _seg((1.85, a), (2.5, a))], 2.5
    if name == "chave":
        return [_seg((0.0, a), (0.8, a)), _dot(0.8, a, 0.06), _seg((0.8, a), (1.68, a - 0.58)),
                _dot(1.75, a, 0.06), _seg((1.75, a), (2.55, a))], 2.55
    if name in ("amperímetro", "voltímetro"):
        c = 1.25
        if name == "amperímetro":
            letter = [_poly((c - 0.26, a + 0.32), (c, a - 0.36), (c + 0.26, a + 0.32)), _seg((c - 0.14, a + 0.06), (c + 0.14, a + 0.06), 4)]
        else:
            letter = [_poly((c - 0.25, a - 0.34), (c, a + 0.32), (c + 0.25, a - 0.34))]
        return [_seg((0.0, a), (0.62, a)), _oval(c, a, 0.63, 0.63)] + letter + [_seg((1.88, a), (2.5, a))], 2.5
    if name == "fio":
        return [_seg((0.0, a), (1.5, a))], 1.5
    if name == "fonte corrente":
        c = 1.25
        return [_seg((0.0, a), (0.62, a)), _oval(c, a, 0.63, 0.63), _seg((c - 0.38, a), (c + 0.32, a)),
                _poly((c + 0.12, a - 0.2), (c + 0.36, a), (c + 0.12, a + 0.2)), _seg((1.88, a), (2.5, a))], 2.5
    if name == "fusível":
        return [_seg((0.0, a), (0.6, a)), _poly((0.6, a - 0.25), (1.9, a - 0.25), (1.9, a + 0.25), (0.6, a + 0.25), (0.6, a - 0.25)),
                _seg((0.6, a), (1.9, a)), _seg((1.9, a), (2.5, a))], 2.5
    if name == "motor":
        c = 1.25
        m = [_poly((c - 0.3, a + 0.3), (c - 0.3, a - 0.3), (c, a + 0.05), (c + 0.3, a - 0.3), (c + 0.3, a + 0.3), n=4)]
        return [_seg((0.0, a), (0.62, a)), _oval(c, a, 0.63, 0.63)] + m + [_seg((1.88, a), (2.5, a))], 2.5
    if name == "ohmímetro":
        c = 1.25
        om = [np.vstack([_seg((c - 0.32, a + 0.3), (c - 0.12, a + 0.3), 3),
                         _S((c - 0.12, a + 0.3), (c - 0.12, a + 0.2), (c - 0.28, a), (c - 0.2, a - 0.28), (c, a - 0.36),
                            (c + 0.2, a - 0.28), (c + 0.28, a), (c + 0.12, a + 0.2), (c + 0.12, a + 0.3)),
                         _seg((c + 0.12, a + 0.3), (c + 0.32, a + 0.3), 3)])]
        return [_seg((0.0, a), (0.62, a)), _oval(c, a, 0.63, 0.63)] + om + [_seg((1.88, a), (2.5, a))], 2.5
    if name == "potenciômetro":
        s_, w = component("resistor")
        return s_ + [_seg((0.6, a + 0.6), (2.0, a - 0.62)), _poly((1.72, a - 0.62), (2.0, a - 0.62), (2.0, a - 0.35), n=3)], w
    if name == "chave fechada":
        return [_seg((0.0, a), (0.8, a)), _dot(0.8, a, 0.06), _seg((0.8, a), (1.76, a - 0.08)),
                _dot(1.75, a, 0.06), _seg((1.75, a), (2.55, a))], 2.55
    if name == "impedância":
        return [_seg((0.0, a), (0.5, a)), _poly((0.5, a - 0.3), (2.1, a - 0.3), (2.1, a + 0.3), (0.5, a + 0.3), (0.5, a - 0.3)),
                _seg((2.1, a), (2.6, a))], 2.6
    return None


SUPPORTED = set("−✓✔✗✘•●().:;,+=≈≠×*÷·/\\<>≤≥%°[]{}|→←~_√±∞π@&$€") | set(GREEK_SHAPES)


def pen(strokes: list[np.ndarray], rng: np.random.Generator, amp: float = 0.02) -> list[np.ndarray]:
    """Deixa o desenho com cara de caneta: densifica e treme levemente."""
    out = []
    for s in strokes:
        s = np.asarray(s, float)
        if len(s) >= 2:
            seg = np.linalg.norm(np.diff(s, axis=0), axis=1)
            d = np.concatenate([[0], np.cumsum(seg)])
            n = max(int(d[-1] / 0.05), 6)
            t = np.linspace(0, d[-1], n)
            s = np.stack([np.interp(t, d, s[:, 0]), np.interp(t, d, s[:, 1])], 1)
            noise = rng.normal(0, amp, s.shape)
            k = np.ones(9) / 9  # ruído suave, não serrilhado
            noise = np.pad(noise, ((4, 4), (0, 0)), mode="reflect") if len(noise) > 4 else noise
            noise = np.stack([np.convolve(noise[:, i], k, mode="valid" if len(noise) > 8 else "same")[:len(s)]
                              for i in range(2)], 1) * 2.5
            s = s + noise
        out.append(s)
    return out


def hand_line(p0, p1, xh: float, rng: np.random.Generator) -> np.ndarray:
    """Linha reta feita à mão (grade de tabela, traço de fração), em pixels."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    n = max(int(L / 6), 9)
    t = np.linspace(0, 1, n)[:, None]
    pts = p0 * (1 - t) + p1 * t
    normal = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) / max(L, 1e-6)
    bow = np.sin(np.pi * t[:, 0]) * rng.normal(0, 0.06) * xh          # leve barriga
    wav = np.convolve(rng.normal(0, 0.05 * xh, n), np.ones(7) / 7, mode="same")  # tremido
    return pts + normal * (bow + wav)[:, None]
