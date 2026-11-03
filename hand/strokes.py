"""Utilitários de traço: conversões, suavização, alinhamento, preparação da
amostra do usuário e renderização em SVG (com acentos do português)."""
from __future__ import annotations

import unicodedata

import numpy as np
from scipy.signal import savgol_filter

from .model import CHAR_TO_IDX


# ---------------- conversões básicas ----------------
def offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([np.cumsum(offsets[:, :2], axis=0), offsets[:, 2:3]], axis=1)


def coords_to_offsets(coords: np.ndarray) -> np.ndarray:
    off = np.concatenate([coords[1:, :2] - coords[:-1, :2], coords[1:, 2:3]], axis=1)
    return np.concatenate([np.array([[0.0, 0.0, 1.0]]), off], axis=0)


def split_strokes(coords: np.ndarray) -> list[np.ndarray]:
    parts = np.split(coords, np.where(coords[:, 2] == 1)[0] + 1, axis=0)
    return [p for p in parts if len(p)]


def denoise(coords: np.ndarray) -> np.ndarray:
    out = []
    for st in split_strokes(coords):
        if len(st) >= 7:
            x = savgol_filter(st[:, 0], 7, 3, mode="nearest")
            y = savgol_filter(st[:, 1], 7, 3, mode="nearest")
            st = np.stack([x, y, st[:, 2]], axis=1)
        out.append(st)
    return np.vstack(out)


def align(xy: np.ndarray) -> np.ndarray:
    """Remove inclinação global da linha (regressão linear)."""
    X = np.stack([np.ones(len(xy)), xy[:, 0]], axis=1)
    offset, slope = np.linalg.lstsq(X, xy[:, 1], rcond=None)[0]
    t = np.arctan(slope)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return xy @ R - offset


def normalize(offsets: np.ndarray) -> np.ndarray:
    offsets = offsets.copy()
    med = np.median(np.linalg.norm(offsets[:, :2], axis=1))
    offsets[:, :2] /= max(med, 1e-6)
    return offsets


# ---------------- texto: português -> alfabeto do modelo ----------------
ACCENT_OF = {"́": "acute", "̀": "grave", "̂": "circ", "̃": "tilde", "̈": "trema"}
UPPER_FALLBACK = {"Q": "q", "X": "x", "Z": "z"}  # maiúsculas que o modelo não conhece
SYMBOL_FALLBACK = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...", "º": "o", "ª": "a"}


def to_model_text(text: str):
    """Converte texto em português para o alfabeto do modelo.

    Retorna (texto_base, marcas) onde marcas[i] é o diacrítico a desenhar sobre
    (ou sob) o caractere i do texto_base: 'acute', 'grave', 'circ', 'tilde',
    'trema' ou 'cedilla'.
    """
    base, marks = [], []
    for ch in text:
        for a, b in SYMBOL_FALLBACK.items():
            if ch == a:
                ch = b
        for sub in ch:
            decomp = unicodedata.normalize("NFD", sub)
            letter, mark = decomp[0], None
            for comb in decomp[1:]:
                if comb == "̧":
                    mark = "cedilla"
                elif comb in ACCENT_OF:
                    mark = ACCENT_OF[comb]
            letter = UPPER_FALLBACK.get(letter, letter)
            if letter not in CHAR_TO_IDX:
                letter = " " if letter.isspace() else ""
            if not letter:
                continue
            base.append(letter)
            marks.append(mark)
    return "".join(base), marks


def wrap_text(text: str, max_chars: int) -> list[str]:
    """Quebra parágrafos em linhas de no máximo max_chars caracteres."""
    lines = []
    for para in text.replace("\r", "").split("\n"):
        words, cur = para.split(), ""
        if not words:
            lines.append("")
            continue
        for w in words:
            while len(w) > max_chars:
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(w[:max_chars - 1] + "-")
                w = w[max_chars - 1:]
            if not cur:
                cur = w
            elif len(cur) + 1 + len(w) <= max_chars:
                cur += " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    return lines


# ---------------- amostra do usuário -> priming ----------------
POINTS_PER_CHAR = 28.0  # densidade média dos dados de treino (IAM On-Line)
ACCENT_LAG = 8  # atraso (em pontos) entre a atenção e a caneta, medido empiricamente


def prepare_user_sample(strokes: list[list[list[float]]], n_chars: int,
                        points_per_char: float = POINTS_PER_CHAR) -> np.ndarray:
    """Converte traços do navegador para o formato do modelo.

    strokes: lista de traços, cada um [[x, y, t_ms], ...] em pixels (y para baixo).
    O navegador entrega pontos em ritmo irregular; a mesa digitalizadora dos
    dados de treino amostrava em ritmo fixo. Então reamostramos cada traço em
    intervalos de tempo iguais (preserva onde a pessoa acelera/desacelera),
    escolhendo o intervalo para dar ~28 pontos por caractere, como no treino.
    Sem tempo, cai para reamostragem por distância.
    """
    strokes = [np.asarray(s, dtype=np.float64) for s in strokes if len(s) >= 1]
    if not strokes:
        raise ValueError("amostra vazia")
    n_target = points_per_char * max(n_chars, 1)
    has_time = all(s.shape[1] >= 3 for s in strokes)
    if has_time:
        durations = [max(s[-1, 2] - s[0, 2], 0.0) for s in strokes]
        total = sum(durations)
        has_time = total > 50  # ms
    if has_time:
        dt = total / n_target
        res = [_resample_time(s, dt) for s in strokes]
    else:
        lengths = [np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1).sum() if len(s) > 1 else 0 for s in strokes]
        step = max(sum(lengths) / n_target, 1e-6)
        res = [_resample(s[:, :2], step) for s in strokes]

    coords = []
    for s in res:
        s = s.copy()
        s[:, 1] *= -1
        eos = np.zeros((len(s), 1))
        eos[-1] = 1
        coords.append(np.hstack([s, eos]))
    coords = np.vstack(coords)
    coords[:, :2] = align(coords[:, :2])
    coords = denoise(coords)
    off = coords_to_offsets(coords)[:1200]
    return normalize(off).astype(np.float32)


def _resample_time(s: np.ndarray, dt: float) -> np.ndarray:
    t = s[:, 2] - s[0, 2]
    # remove timestamps repetidos/fora de ordem
    keep = np.concatenate([[True], np.diff(t) > 0])
    s, t = s[keep], t[keep]
    if len(s) < 2 or t[-1] < dt:
        p = s[:, :2]
        return np.vstack([p[0], p[-1] + 1e-3]) if len(p) else p
    tt = np.arange(0, t[-1], dt)
    if t[-1] - tt[-1] > dt * 0.3:
        tt = np.append(tt, t[-1])
    return np.stack([np.interp(tt, t, s[:, 0]), np.interp(tt, t, s[:, 1])], axis=1)


def _resample(pts: np.ndarray, step: float) -> np.ndarray:
    if len(pts) < 2:
        return np.vstack([pts, pts + 1e-3])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    dist = np.concatenate([[0], np.cumsum(seg)])
    total = dist[-1]
    if total < step:
        return np.vstack([pts[0], pts[-1] + 1e-3])
    t = np.arange(0, total, step)
    if total - t[-1] > step * 0.3:
        t = np.append(t, total)
    return np.stack([np.interp(t, dist, pts[:, 0]), np.interp(t, dist, pts[:, 1])], axis=1)


# ---------------- renderização ----------------
def line_coords(offsets: np.ndarray, scale: float = 1.5) -> np.ndarray:
    """Deslocamentos -> coordenadas de tela (y para baixo), alinhadas."""
    off = offsets.copy()
    off[:, :2] *= scale
    c = offsets_to_coords(off)
    c = denoise(c)
    c[:, :2] = align(c[:, :2])
    c[:, 1] *= -1
    return c


def _path_d(pts: np.ndarray) -> str:
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)


def _diacritic_paths(mark: str, cx: float, top: float, bottom: float, size: float, rng) -> list[str]:
    j = lambda: rng.normal(0, size * 0.06)
    s = size
    y = top - s * 0.35
    if mark == "acute":
        return [_path_d(np.array([[cx - s * .12 + j(), y + s * .18], [cx + s * .2 + j(), y - s * .22]]))]
    if mark == "grave":
        return [_path_d(np.array([[cx - s * .2 + j(), y - s * .22], [cx + s * .12 + j(), y + s * .18]]))]
    if mark == "circ":
        return [_path_d(np.array([[cx - s * .25, y + s * .12], [cx + j(), y - s * .2], [cx + s * .25, y + s * .12]]))]
    if mark == "tilde":
        xs = np.linspace(-.3, .3, 9)
        ys = -np.sin(xs / .3 * np.pi) * .1
        return [_path_d(np.stack([cx + xs * s + j(), y + ys * s], 1))]
    if mark == "trema":
        return [_path_d(np.array([[cx - s * .15, y], [cx - s * .13, y + .5]])),
                _path_d(np.array([[cx + s * .15, y], [cx + s * .17, y + .5]]))]
    if mark == "cedilla":
        b = bottom + s * .05
        return [_path_d(np.array([[cx, b], [cx - s * .02, b + s * .15], [cx + s * .14, b + s * .22],
                                  [cx + s * .06, b + s * .38], [cx - s * .12, b + s * .36]]))]
    return []


def render_page(lines: list[dict], *, width: float = 900, line_height: float = 64, margin_left: float = 70,
                margin_top: float = 80, paper: str = "pautado", ink: str = "#1b2a6b", stroke_width: float = 2.2,
                seed: int | None = None, min_lines: int = 0) -> str:
    """Monta a página SVG. lines: [{'coords': (T,3), 'char_idx': (T,), 'marks': [...]}] (ou {} p/ linha vazia)."""
    rng = np.random.default_rng(seed)
    n_rows = max(len(lines), min_lines)
    height = margin_top + line_height * (n_rows + 1)
    bg = []
    if paper != "branco":
        if paper == "pautado":
            for i in range(n_rows + 1):
                yy = margin_top + line_height * i + line_height * 0.22
                bg.append(f'<line x1="0" y1="{yy:.1f}" x2="{width}" y2="{yy:.1f}" stroke="#9cc3e6" stroke-width="1"/>')
            bg.append(f'<line x1="{margin_left - 16}" y1="0" x2="{margin_left - 16}" y2="{height}" stroke="#e9a3a3" stroke-width="1.2"/>')
        elif paper == "quadriculado":
            g = line_height / 3
            for i in range(int(width / g) + 1):
                bg.append(f'<line x1="{i * g:.1f}" y1="0" x2="{i * g:.1f}" y2="{height}" stroke="#d6e4f0" stroke-width=".8"/>')
            for i in range(int(height / g) + 1):
                bg.append(f'<line x1="0" y1="{i * g:.1f}" x2="{width}" y2="{i * g:.1f}" stroke="#d6e4f0" stroke-width=".8"/>')

    groups = []
    for li, ln in enumerate(lines):
        if not ln:
            continue
        paths = line_paths(ln, li, width=width, line_height=line_height, margin_left=margin_left,
                           margin_top=margin_top, rng=rng)
        groups.append(line_group(li, paths, width=width, line_height=line_height, margin_top=margin_top))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" width="{width}" height="{height:.0f}">'
        f'<rect width="100%" height="100%" fill="#fffdf7"/>{"".join(bg)}'
        f'<g fill="none" stroke="{ink}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
        f'{"".join(groups)}</g></svg>'
    )


def line_group(li: int, paths: list[str], *, width: float = 900, line_height: float = 64,
               margin_top: float = 80) -> str:
    """Um grupo SVG por linha, com uma área clicável (para reescrever a linha)."""
    baseline_y = margin_top + line_height * li + line_height * 0.22
    hit = (f'<rect class="hit" x="0" y="{baseline_y - line_height * 0.8:.1f}" width="{width}" '
           f'height="{line_height:.1f}" fill="transparent" stroke="none"/>')
    body = "".join(f'<path d="{d}"/>' for d in paths)
    return f'<g class="linha" data-line="{li}">{hit}{body}</g>'


def line_paths(ln: dict, li: int, *, width: float = 900, line_height: float = 64, margin_left: float = 70,
               margin_top: float = 80, rng=None) -> list[str]:
    """Caminhos SVG (atributo d) de uma linha escrita, já na posição da pauta."""
    rng = rng if rng is not None else np.random.default_rng()
    paths: list[str] = []
    c, cidx, marks = ln["coords"], ln["char_idx"], ln.get("marks") or []
    baseline_y = margin_top + line_height * li + line_height * 0.22
    xy = c[:, :2].copy()
    # base da escrita ~ percentil 88 do y (y para baixo), encaixa na pauta
    xy[:, 1] += baseline_y - np.percentile(xy[:, 1], 88)
    xy[:, 0] += margin_left - xy[:, 0].min()
    max_w = width - margin_left - 20
    cur_w = xy[:, 0].max() - margin_left
    if cur_w > max_w:  # encolhe se passar da margem
        f = max_w / cur_w
        xy[:, 0] = margin_left + (xy[:, 0] - margin_left) * f
    for st in split_strokes(np.hstack([xy, c[:, 2:3]])):
        if len(st) == 1:
            st = np.vstack([st, st + [0.6, 0.6, 0]])
        paths.append(_path_d(st[:, :2]))

    # acentos: a atenção do modelo diz em qual letra ele está em cada ponto.
    # Ela "pula" para a próxima letra ~8 pontos antes da caneta terminar a
    # atual, então atrasamos os rótulos antes de medir a posição da letra.
    size = line_height * 0.55
    lag = ACCENT_LAG
    lab = np.concatenate([np.full(lag, cidx[0]), cidx[:len(cidx) - lag]]) if len(cidx) > lag else cidx
    for k, mark in enumerate(marks):
        if not mark:
            continue
        sel = np.where(lab == k)[0]
        if len(sel) < 2:
            continue
        x0, x1 = np.percentile(xy[sel, 0], [10, 90])
        cx = (x0 + x1) / 2
        near = np.abs(xy[:, 0] - cx) <= max((x1 - x0) * 0.35, 2.0)
        pts = xy[near] if near.sum() >= 2 else xy[sel]
        top = float(np.percentile(pts[:, 1], 3))
        bottom = float(np.percentile(xy[sel, 1], 92))
        paths.extend(_diacritic_paths(mark, cx, top, bottom, size, rng))
    return paths
