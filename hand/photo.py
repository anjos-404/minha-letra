"""Foto de caderno -> traços de caneta (aproximados).

O modelo aprende pelo caminho da caneta, e a foto só tem a tinta. Então:
  1. separa tinta de papel com limiar local (aguenta sombra e luz desigual);
  2. remove a pauta do caderno (linhas horizontais longas e finas);
  3. limpa sujeira e recorta a linha escrita;
  4. afina a tinta até virar um fio de 1 pixel (Zhang-Suen);
  5. percorre o fio como se fosse a caneta: da esquerda para a direita,
     pingos e cortes de "t" logo depois da letra a que pertencem.

Só usa NumPy, SciPy e Pillow.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi

TARGET_HEIGHT = 110  # altura da linha escrita depois de normalizar (px)


# ---------------- entrada ----------------
def load_image(data: bytes | str) -> np.ndarray:
    """Bytes ou data URL -> imagem em tons de cinza [0,1] (0 = preto)."""
    if isinstance(data, str):
        if "," in data[:100]:
            data = data.split(",", 1)[1]
        data = base64.b64decode(data)
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > 2400:
        img.thumbnail((2400, 2400))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    return _ink_gray(rgb)


def _ink_gray(rgb: np.ndarray) -> np.ndarray:
    """Cinza que realça a tinta e apaga a pauta clara.

    Usa o canal mais escuro (tinta azul, preta ou vermelha some menos que o
    papel em algum canal), mas as linhas de caderno são claras e coloridas,
    então ficam perto do branco no canal de luminância.
    """
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    darkest = rgb.min(axis=2)
    # tinta: escura em algum canal E com luminância não muito alta
    return np.minimum(lum, 0.5 * (lum + darkest))


# ---------------- binarização ----------------
def _sauvola(gray: np.ndarray, window: int, k: float = 0.25, R: float = 0.5) -> np.ndarray:
    w = max(3, window | 1)
    mean = ndi.uniform_filter(gray, w)
    sq = ndi.uniform_filter(gray * gray, w)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thr = mean * (1 + k * (std / R - 1))
    return gray < thr


def _remove_rules(ink: np.ndarray) -> np.ndarray:
    """Remove linhas horizontais longas (pauta) e a margem vertical."""
    h, w = ink.shape
    run = max(int(w * 0.18), 25)
    horiz = ndi.binary_opening(ink, structure=np.ones((1, run), bool))
    vrun = max(int(h * 0.6), 25)
    vert = ndi.binary_opening(ink, structure=np.ones((vrun, 1), bool))
    lines = ndi.binary_dilation(horiz | vert, structure=np.ones((3, 3), bool))
    cleaned = ink & ~lines
    # repara pequenos cortes que a remoção da pauta abriu nas letras
    bridged = ndi.binary_closing(cleaned, structure=np.ones((5, 1), bool))
    return cleaned | (bridged & lines)


def _clean(ink: np.ndarray, min_area: int) -> np.ndarray:
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n == 0:
        return ink
    areas = ndi.sum(ink, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = areas >= min_area
    return keep[lab]


def binarize(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    ink = _sauvola(gray, window=max(15, int(min(h, w) * 0.12)))
    # papel muito claro nunca é tinta (evita ruído em áreas lisas)
    ink &= gray < np.percentile(gray, 60) * 0.92
    ink = _remove_rules(ink)
    return _clean(ink, min_area=max(6, int(h * w * 2e-5)))


def select_line(ink: np.ndarray, y_top: float, y_bottom: float) -> np.ndarray:
    """Fica só com as manchas de tinta da linha marcada pela pessoa.

    Uma mancha entra se o centro dela está na faixa marcada ou se a maior
    parte dela está lá. Assim a perninha do g/p/q entra inteira mesmo saindo
    da faixa, e a escrita das linhas de cima e de baixo fica de fora.
    """
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n == 0:
        return ink
    info = []  # (id, centro, altura, área, fração dentro da faixa)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        ys = np.nonzero(lab[sl] == i)[0] + sl[0].start
        info.append((i, (ys.min() + ys.max()) / 2, ys.max() - ys.min() + 1, len(ys),
                     np.mean((ys >= y_top) & (ys <= y_bottom))))
    core = [c for c in info if y_top <= c[1] <= y_bottom or c[4] >= 0.5]
    keep = np.zeros(n + 1, bool)
    if not core:
        return ink & False
    # centro real da linha escrita = média dos centros das manchas grandes
    # que caíram na faixa (a foto pode estar torta ou a seleção deslocada)
    big = [c for c in core if c[3] >= np.median([c[3] for c in core])] or core
    w = np.array([c[3] for c in big], float)
    y_c = float(np.sum(w * np.array([c[1] for c in big])) / w.sum())
    h_m = float(np.median([c[2] for c in big]))
    tol = max(0.55 * h_m, 0.3 * (y_bottom - y_top))
    for c in info:
        near = abs(c[1] - y_c) <= tol
        keep[c[0]] = near or (c[4] >= 0.5 and abs(c[1] - y_c) <= 1.5 * tol)
    return keep[lab]


def _ink_box(ink: np.ndarray, shape, pad: float = 0.08):
    ys, xs = np.nonzero(ink)
    if len(xs) == 0:
        raise ValueError("Não encontrei tinta na foto. Tente uma foto mais nítida e com boa luz.")
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ph, pw = int((y1 - y0) * pad) + 2, int((x1 - x0) * 0.02) + 2
    return (max(0, y0 - ph), min(shape[0], y1 + ph + 1), max(0, x0 - pw), min(shape[1], x1 + pw + 1))


def touches_edge(ink: np.ndarray, margin: int = 2) -> bool:
    """Tinta encostando em cima/embaixo do recorte = letra provavelmente cortada."""
    return bool(ink[:margin].any() or ink[-margin:].any())


# ---------------- afinamento (Zhang-Suen) ----------------
def thin(img: np.ndarray) -> np.ndarray:
    im = np.pad(img.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = [np.roll(np.roll(im, dy, 0), dx, 1) for dy, dx in
                 ((1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1))]
            # P2..P9 no sentido horário a partir de cima
            p2, p3, p4, p5, p6, p7, p8, p9 = P
            B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            A = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (im == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rm.any():
                im[rm] = 0
                changed = True
        if not changed:
            break
    return im[1:-1, 1:-1].astype(bool)


def _neighbors_count(sk: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    return ndi.convolve(sk.astype(int), k, mode="constant") * sk


def prune_spurs(skel: np.ndarray, max_len: int = 6) -> np.ndarray:
    """Remove rebarbas: galhos curtos que saem de um cruzamento e terminam
    em ponta. O afinamento cria isso nas junções e nas pontas grossas."""
    sk = skel.copy()
    for _ in range(2):
        nc = _neighbors_count(sk)
        ends = list(zip(*np.nonzero((nc == 1))))
        removed = False
        for e in ends:
            path = [e]
            prev, cur = None, e
            hit_junction = False
            while len(path) <= max_len:
                y, x = cur
                nb = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                      if (dy or dx) and 0 <= y + dy < sk.shape[0] and 0 <= x + dx < sk.shape[1]
                      and sk[y + dy, x + dx] and (y + dy, x + dx) != prev and (y + dy, x + dx) not in path]
                if not nb:
                    break
                if len(nb) > 1 or nc[nb[0]] >= 3:
                    hit_junction = True
                    break
                prev, cur = cur, nb[0]
                path.append(cur)
            if hit_junction and len(path) <= max_len:
                for p in path:
                    sk[p] = False
                removed = True
        if not removed:
            break
    return sk


# ---------------- esqueleto -> traços ----------------
_NB = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]  # 4-vizinhos primeiro


def trace(skel: np.ndarray) -> list[np.ndarray]:
    ys, xs = np.nonzero(skel)
    pix = set(zip(ys.tolist(), xs.tolist()))

    def nbrs(p):
        y, x = p
        return [(y + dy, x + dx) for dy, dx in _NB if (y + dy, x + dx) in pix]

    deg = {p: len(nbrs(p)) for p in pix}
    visited: set = set()
    strokes = []
    endpoints = sorted((p for p in pix if deg[p] == 1), key=lambda p: (p[1], p[0]))
    rest = sorted(pix, key=lambda p: (p[1], p[0]))
    for start in endpoints + rest:
        if start in visited:
            continue
        path = [start]
        visited.add(start)
        cur, d = start, None
        while True:
            cand = [n for n in nbrs(cur) if n not in visited]
            if not cand:
                break
            if d is not None and len(cand) > 1:
                cand.sort(key=lambda n: -((n[0] - cur[0]) * d[0] + (n[1] - cur[1]) * d[1]))
            nxt = cand[0]
            # pixels "de canto" vizinhos dos dois: marcados para não virarem lixo
            for n in cand[1:]:
                if abs(n[0] - nxt[0]) <= 1 and abs(n[1] - nxt[1]) <= 1 and deg.get(n, 0) <= 2:
                    visited.add(n)
            d = (nxt[0] - cur[0], nxt[1] - cur[1])
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        strokes.append(np.array([[x, y] for y, x in path], dtype=np.float64))
    return strokes


def _length(s: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) if len(s) > 1 else 0.0


def _drop_isolated_bits(strokes: list[np.ndarray], height: float) -> list[np.ndarray]:
    """Caquinhos soltos longe de tudo (resto de margem, sujeira) saem."""
    if len(strokes) < 3:
        return strokes
    keep = []
    for i, s in enumerate(strokes):
        if _length(s) >= height * 0.15:
            keep.append(s)
            continue
        x0, x1 = s[:, 0].min(), s[:, 0].max()
        gap = min((max(o[:, 0].min() - x1, x0 - o[:, 0].max(), 0.0) for j, o in enumerate(strokes) if j != i),
                  default=0.0)
        if gap <= height * 0.3:
            keep.append(s)
    return keep


def _merge_and_order(strokes: list[np.ndarray], height: float, *, min_len: float = 3.0,
                     orient: bool = True, join_dist: float = 2.5, small_after: bool = True) -> list[np.ndarray]:
    strokes = [s for s in strokes if _length(s) >= min_len or len(strokes) < 3]
    strokes = _drop_isolated_bits(strokes, height)
    # orienta cada traço da esquerda para a direita (como a mão costuma ir)
    if orient:
        strokes = [s[::-1] if s[0, 0] - s[-1, 0] > height * 0.15 else s for s in strokes]
    strokes.sort(key=lambda s: s[:, 0].min())

    # junta pedaços cujo fim encosta no começo do outro (o afinamento quebra junções)
    merged: list[np.ndarray] = []
    for s in strokes:
        joined = False
        for i in range(len(merged) - 1, max(-1, len(merged) - 6), -1):
            m = merged[i]
            if np.linalg.norm(m[-1] - s[0]) <= join_dist:
                merged[i] = np.vstack([m, s])
                joined = True
                break
            if np.linalg.norm(m[-1] - s[-1]) <= join_dist:
                merged[i] = np.vstack([m, s[::-1]])
                joined = True
                break
        if not joined:
            merged.append(s)

    if not small_after:
        return sorted(merged, key=lambda s: s[:, 0].min())

    # traços pequenos (pingo do i, corte do t, acento) vão logo depois do traço
    # grande que está embaixo deles, como acontece quando a gente escreve
    small_cut = height * 0.35
    big = [s for s in merged if _length(s) >= small_cut]
    small = [s for s in merged if _length(s) < small_cut]
    big.sort(key=lambda s: s[:, 0].min())
    after: dict[int, list[np.ndarray]] = {i: [] for i in range(len(big))}
    lead: list[np.ndarray] = []
    for s in small:
        cx = s[:, 0].mean()
        owner = None
        for i, b in enumerate(big):
            if b[:, 0].min() - 2 <= cx <= b[:, 0].max() + 2:
                owner = i
        if owner is None:
            owner = max((i for i, b in enumerate(big) if b[:, 0].min() <= cx), default=None)
        (after[owner] if owner is not None else lead).append(s)
    out = lead[:]
    for i, b in enumerate(big):
        out.append(b)
        out.extend(sorted(after[i], key=lambda s: s[:, 0].min()))
    return out


def _resize(a: np.ndarray, size) -> np.ndarray:
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize(size, Image.LANCZOS),
                      dtype=np.float32) / 255


def photo_to_strokes(data: bytes | str, focus: tuple[float, float] | None = None) -> dict:
    """Retorna {'strokes': [[[x,y],...],...], 'width', 'height', 'clipped'} em pixels.

    focus: (topo, base) da linha marcada pela pessoa, em fração da altura da
    imagem enviada (a imagem vem com folga em volta da seleção).
    """
    gray = load_image(data)
    # 1ª passada: acha a linha escrita e recorta
    h, w = gray.shape
    scale = min(1.0, 1600 / max(h, w))
    if scale < 1:
        gray = _resize(gray, (int(w * scale), int(h * scale)))
        h, w = gray.shape
    ink1 = binarize(gray)
    if focus is not None:
        ink1 = select_line(ink1, focus[0] * h, focus[1] * h)
        ink1 = _clean(ink1, min_area=max(6, int(h * w * 2e-5)))
    clipped = touches_edge(ink1)
    y0, y1, x0, x1 = _ink_box(ink1, gray.shape)
    crop = gray[y0:y1, x0:x1]
    keep_mask = ink1[y0:y1, x0:x1]

    # 2ª passada: normaliza a altura para o afinamento ficar estável
    ch, cw = crop.shape
    f = TARGET_HEIGHT / max(ch, 1)
    size = (max(8, int(cw * f)), TARGET_HEIGHT)
    crop = _resize(crop, size)
    ink = binarize(crop)
    # mantém só as manchas que já eram da linha na 1ª passada
    mask = _resize(keep_mask.astype(np.float32), size) > 0.2
    mask = ndi.binary_dilation(mask, structure=np.ones((5, 5), bool))
    lab, n = ndi.label(ink, structure=np.ones((3, 3), bool))
    if n:
        hits = ndi.maximum(mask, lab, index=np.arange(1, n + 1))
        ok = np.zeros(n + 1, bool)
        ok[1:] = np.asarray(hits, bool)
        ink = ok[lab]
    ink = ndi.binary_closing(ink, structure=np.ones((3, 3), bool))
    ink = ndi.binary_fill_holes(ink) & ~_small_holes_only(ink)
    skel = prune_spurs(thin(ink), max_len=int(TARGET_HEIGHT * 0.06))
    raw = trace(skel)
    # Várias reconstruções possíveis do mesmo desenho; quem escolhe a mais
    # natural é o modelo (HandwritingModel.judge_samples), não uma regra fixa.
    variants = []
    for orient in (True, False):
        for join_dist in (2.5, 5.0):
            for small_after in (True, False):
                for min_len in (3.0, 8.0):
                    v = _merge_and_order(list(raw), TARGET_HEIGHT, min_len=min_len, orient=orient,
                                         join_dist=join_dist, small_after=small_after)
                    if v:
                        variants.append([s.round(1).tolist() for s in v])
    if not variants:
        raise ValueError("Não consegui ler traços na foto. Recorte só a linha escrita.")
    strokes = variants[0]
    return {
        "variants": variants,
        "strokes": strokes,
        "width": int(ink.shape[1]),
        "height": int(ink.shape[0]),
        "clipped": clipped,
    }


def _small_holes_only(ink: np.ndarray) -> np.ndarray:
    """Buracos grandes (miolo do 'o', 'a', 'e') são devolvidos depois do fill;
    só buracos minúsculos (falhas da caneta) ficam preenchidos."""
    filled = ndi.binary_fill_holes(ink)
    holes = filled & ~ink
    lab, n = ndi.label(holes)
    if n == 0:
        return np.zeros_like(ink)
    areas = ndi.sum(holes, lab, index=np.arange(1, n + 1))
    big = np.zeros(n + 1, bool)
    big[1:] = areas > 6
    return big[lab]
