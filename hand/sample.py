"""Escolhe, entre várias reconstruções de uma foto, a que o modelo acha mais natural."""
from __future__ import annotations

import numpy as np

from .model import HandwritingModel
from .strokes import prepare_user_sample, to_model_text


def _norm_points(strokes_xy: list[np.ndarray]) -> np.ndarray:
    """Nuvem de pontos com a mesma escala: altura do corpo da letra = 1."""
    pts = np.vstack(strokes_xy)
    y_lo, y_hi = np.percentile(pts[:, 1], [10, 90])
    s = max(y_hi - y_lo, 1e-6)
    out = (pts - [pts[:, 0].min(), np.median(pts[:, 1])]) / s
    return out


def _resample_cloud(pts_list: list[np.ndarray], step: float) -> list[np.ndarray]:
    out = []
    for p in pts_list:
        if len(p) < 2:
            out.append(p)
            continue
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        d = np.concatenate([[0], np.cumsum(seg)])
        t = np.arange(0, d[-1] + 1e-9, step)
        out.append(np.stack([np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])], 1))
    return out


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Distância média entre duas nuvens de pontos (nos dois sentidos)."""
    from scipy.spatial import cKDTree
    da, _ = cKDTree(b).query(a)
    db, _ = cKDTree(a).query(b)
    return float(np.mean(da) + np.mean(db)) / 2


def photo_cloud(strokes: list) -> np.ndarray:
    """Pontos da foto (y para baixo) na escala normalizada."""
    arrs = [np.asarray(s, float)[:, :2] * [1, -1] for s in strokes if len(s)]  # y para cima, como o modelo
    return _norm_points(_resample_cloud(arrs, _body_height(arrs) * 0.04))


def _body_height(arrs: list[np.ndarray]) -> float:
    pts = np.vstack(arrs)
    lo, hi = np.percentile(pts[:, 1], [10, 90])
    return max(hi - lo, 1e-6)


def offsets_cloud(off: np.ndarray) -> np.ndarray:
    from .strokes import denoise, offsets_to_coords, split_strokes
    c = denoise(offsets_to_coords(off))
    arrs = [st[:, :2] for st in split_strokes(c)]
    return _norm_points(_resample_cloud(arrs, _body_height(arrs) * 0.04))


def distill_style(model: HandwritingModel, photo_strokes, phrase: str, candidates: int = 10,
                  bias: float = 0.9, seed: int | None = None, rounds: int = 3) -> tuple[np.ndarray, str, dict]:
    """Até `rounds` rodadas de reescritas; para na primeira que achar uma boa.

    photo_strokes pode ser uma reconstrução ou uma lista delas, da mais para a
    menos natural: cada rodada tenta a próxima (falham em sorteios diferentes).
    """
    variants = photo_strokes if (photo_strokes and isinstance(photo_strokes[0], list)
                                 and photo_strokes[0] and isinstance(photo_strokes[0][0], list)
                                 and photo_strokes[0][0] and isinstance(photo_strokes[0][0][0], list)) else [photo_strokes]
    info = {}
    for r in range(rounds):
        v = variants[min(r, len(variants) - 1)]
        s = None if seed is None else seed + 101 * r
        off, text, info = _distill_round(model, v, phrase, candidates, bias, s)
        if info.get("distilled"):
            info["round"] = r + 1
            return off, text, info
    # nenhuma reescrita boa: devolve a reconstrução mais natural, sem destilar
    return prepare_user_sample(variants[0], n_chars=len(to_model_text(phrase.strip())[0])), \
        to_model_text(phrase.strip())[0], info


def _distill_round(model: HandwritingModel, photo_strokes: list, phrase: str, candidates: int,
                   bias: float, seed: int | None) -> tuple[np.ndarray, str, dict]:
    """Troca a amostra reconstruída da foto por uma reescrita da mesma frase
    feita pelo próprio modelo, escolhendo a reescrita mais parecida com a foto.

    A reconstrução da foto tem o desenho certo, mas o movimento de caneta é
    adivinhado; o modelo lê isso meio "confuso" e escreve com menos clareza.
    A reescrita escolhida tem o movimento natural dele e a cara da sua letra.
    Retorna (amostra em deslocamentos, texto da amostra, informações).
    """
    base, _ = to_model_text(phrase.strip())
    prime = prepare_user_sample(photo_strokes, n_chars=len(base))
    target = photo_cloud(photo_strokes)
    outs = model.generate([base] * candidates, bias=bias, prime_strokes=prime, prime_text=base, seed=seed)
    outs = [(o, ci) for o, ci in outs if len(o) >= 10]
    if not outs:
        return prime, base, {"distilled": False}
    offs = [o for o, _ in outs]
    margins = legibility(model, offs, base, seed=seed)
    clouds = [offsets_cloud(o) for o in offs]
    dists = [chamfer(c, target) for c in clouds]
    t_w = float(np.ptp(target[:, 0]))
    widths = [float(np.ptp(c[:, 0])) / max(t_w, 1e-6) for c in clouds]
    quals = [model.line_quality(base, ci, len(o)) for o, ci in outs]
    top = max(margins)
    ok = [i for i in range(len(offs))
          if margins[i] >= max(MIN_LEGIBILITY, 0.85 * top)   # legível
          and 0.75 <= widths[i] <= 1.35                       # mesma largura que a foto (garrancho sai espremido)
          and quals[i] <= 2.0                                 # não pulou nem empacou em letras
          and len(offs[i]) >= 18 * len(base)]                 # escreveu a frase inteira
    if not ok:
        return prime, base, {"distilled": False, "best_legibility": round(top, 3)}
    i = min(ok, key=lambda k: dists[k])
    off = offs[i].copy()
    off[0] = [0.0, 0.0, 1.0]
    return off, base, {"distilled": True, "distance": round(dists[i], 3), "legibility": round(margins[i], 3),
                       "candidates": len(offs)}


MIN_LEGIBILITY = 0.42  # escrita de verdade fica em ~0.5–0.6; garrancho perto de 0


def legibility(model: HandwritingModel, offsets: list[np.ndarray], text: str, seed: int | None = None) -> list[float]:
    """Quanto o texto certo explica os traços melhor que o mesmo texto embaralhado.

    É um "OCR" feito com o próprio modelo gerador: se a escrita é legível, ler
    com o texto certo é bem menos surpreendente do que com as letras trocadas.
    """
    rng = np.random.default_rng(seed)
    chars = list(text)
    shuffled = "".join(rng.permutation(chars))
    if shuffled == text and len(set(chars)) > 1:
        shuffled = text[::-1]
    right = model.judge_samples(offsets, text)
    wrong = model.judge_samples(offsets, shuffled)
    return [w["nll"] - r["nll"] for r, w in zip(right, wrong)]


def pick_best_variant(model: HandwritingModel, variants: list, phrase: str) -> tuple[list, list[dict]]:
    """Retorna (traços escolhidos, notas de todas as versões)."""
    base, _ = to_model_text(phrase.strip())
    seen, uniq = set(), []
    for v in variants:
        key = (len(v), tuple(len(s) for s in v), tuple(tuple(s[0]) for s in v[:5]))
        if key not in seen:
            seen.add(key)
            uniq.append(v)
    if len(uniq) == 1 or len(base) < 4:
        return uniq[0], [], uniq
    primes = [prepare_user_sample(v, n_chars=len(base)) for v in uniq]
    notes = model.judge_samples(primes, base)
    scores = [n["nll"] + 0.3 * n["drift"] for n in notes]
    order = list(np.argsort(scores))
    for n, sc in zip(notes, scores):
        n["score"] = round(sc, 3)
    return uniq[order[0]], notes, [uniq[i] for i in order]
