"""Diagramas de circuito desenhados à mão, a partir de uma descrição em texto.

Linguagem (uma linha começando com "circuito:", ou várias linhas logo abaixo):

    circuito: bateria(12 V) - R1(10 Ω) - (R2(20 Ω) || C1(47 μF)) - terra

    -   em série          ||  (ou //)  em paralelo         ( )  agrupa
    NOME(valor)  componente com rótulo; o tipo vem do nome:
        R, R1, Req, resistor ......... resistor       C, C1 .......... capacitor
        L, L1, indutor, bobina ....... indutor        D, D1, diodo ... diodo
        V, E, Vs, bateria, pilha ..... fonte DC       LED ............ LED
        fonte ac, gerador, Vac, ~ .... fonte AC       lâmpada, lamp .. lâmpada
        I, Is, fonte de corrente ..... fonte corrente chave, S, K .... chave aberta
        chave fechada ................ chave fechada  F, fusível ..... fusível
        A, amperímetro ............... amperímetro    M, motor ....... motor
        voltímetro, Vm ............... voltímetro     ohmímetro ...... ohmímetro
        pot, potenciômetro ........... potenciômetro  Z, Z1 e outros . caixa (impedância)
        fio .......................... fio            terra, GND ..... aterramento

    Com várias linhas, cada linha é um trecho em série:
        circuito:
        bateria(9 V)
        R1(100 Ω)
        R2(200 Ω) || R3(300 Ω)

    Sem fonte, ou com "circuito A-B:", desenha os terminais em aberto
    (bom para exercício de resistência equivalente).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .symbols import AXIS, component, hand_line, pen


# ---------------------------------------------------------------- árvore
@dataclass
class Comp:
    kind: str
    name: str = ""
    value: str = ""
    label: list = field(default_factory=list)   # peças (parse_inline) do rótulo


@dataclass
class Series:
    parts: list


@dataclass
class Parallel:
    branches: list


@dataclass
class Circuit:
    tree: object                     # Series
    source: Comp | None = None       # fonte que vai na lateral esquerda
    ground: bool = False
    terminals: tuple = ()            # ("A", "B") quando o circuito fica aberto
    term_items: list = field(default_factory=list)   # rótulos dos terminais (peças)


SOURCES = {"bateria", "fonte ac", "fonte corrente"}

_WORD_KINDS = [
    (r"^(fonte\s*(ac|ca)|gerador|vac|~)$", "fonte ac"),
    (r"^(fonte\s*de\s*corrente|fonte\s*corrente|is|i\d*)$", "fonte corrente"),
    (r"^(bateria|pilha|fonte(\s*(dc|cc))?|vs|vcc|vdc|v\d*|e\d*|ε\d*)$", "bateria"),
    (r"^(volt[ií]metro|vm\d*)$", "voltímetro"),
    (r"^(amper[ií]metro|a\d*)$", "amperímetro"),
    (r"^(ohm[ií]metro)$", "ohmímetro"),
    (r"^(led\d*)$", "led"),
    (r"^(l[âa]mpada|lamp\d*)$", "lâmpada"),
    (r"^(chave\s*fechada|interruptor\s*fechado)$", "chave fechada"),
    (r"^(chave(\s*aberta)?|interruptor(\s*aberto)?|sw\d*|s\d*|k\d*)$", "chave"),
    (r"^(fus[ií]vel|f\d*)$", "fusível"),
    (r"^(motor|m\d*)$", "motor"),
    (r"^(pot\w*|potenci[ôo]metro)$", "potenciômetro"),
    (r"^(fio)$", "fio"),
    (r"^(terra|gnd|aterramento)$", "terra"),
    (r"^(resistor|resist[êe]ncia|r\w*)$", "resistor"),
    (r"^(capacitor|condensador|c\w*)$", "capacitor"),
    (r"^(indutor|bobina|l\w*)$", "indutor"),
    (r"^(diodo|d\w*)$", "diodo"),
]
# palavras que já dizem o que é: não precisam virar rótulo
_GENERIC = re.compile(r"^(bateria|pilha|fonte.*|gerador|resistor|resist[êe]ncia|capacitor|condensador|indutor|"
                      r"bobina|diodo|l[âa]mpada|chave.*|interruptor.*|fus[ií]vel|motor|volt[ií]metro|"
                      r"amper[ií]metro|ohm[ií]metro|potenci[ôo]metro|fio|terra|gnd)$", re.I)


def kind_of(name: str) -> str:
    n = name.strip().lower()
    for pat, kind in _WORD_KINDS:
        if re.match(pat, n):
            return kind
    return "impedância"


# ---------------------------------------------------------------- leitura
HEADER = re.compile(r"^\s*circuito\s*(?:([A-Za-z0-9]+)\s*[-–]\s*([A-Za-z0-9]+))?\s*:\s*(.*)$", re.I)


def _split(s: str, ops: tuple[str, ...]) -> list[str]:
    out, depth, cur, i = [], 0, [], 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0:
            hit = next((op for op in ops if s.startswith(op, i)), None)
            if hit:
                out.append("".join(cur))
                cur = []
                i += len(hit)
                continue
        cur.append(ch)
        i += 1
    out.append("".join(cur))
    return [p.strip() for p in out]


SERIES_OPS = (" - ", " — ", " – ", "->", "→", " + ")
PAR_OPS = ("||", "//", "∥")


def _series_split(s: str) -> list[str]:
    # aceita "R1-R2" sem espaços também, desde que fora dos parênteses do valor
    s = re.sub(r"(?<=[\w)\]])\s*[-–—]\s*(?=[\w(\[~])", " - ", s)
    return [p for p in _split(s, SERIES_OPS) if p]


def parse_expr(s: str):
    parts = _series_split(s)
    if len(parts) > 1:
        return Series([parse_expr(p) for p in parts])
    branches = [b for b in _split(s, PAR_OPS) if b]
    if len(branches) > 1:
        return Parallel([parse_expr(b) for b in branches])
    s = s.strip()
    if s.startswith("(") and s.endswith(")") and _closes_at_end(s):
        return parse_expr(s[1:-1])
    m = re.match(r"^([^()]+?)\s*(?:\((.*)\))?$", s)
    name, value = (m.group(1).strip(), (m.group(2) or "").strip()) if m else (s, "")
    return Comp(kind_of(name), name, value)


def _closes_at_end(s: str) -> bool:
    depth = 0
    for k, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and k < len(s) - 1:
            return False
    return True


def _flatten(node):
    if isinstance(node, Series):
        out = []
        for p in node.parts:
            q = _flatten(p)
            out.extend(q.parts if isinstance(q, Series) else [q])
        return Series(out)
    if isinstance(node, Parallel):
        return Parallel([_flatten(b) for b in node.branches])
    return node


def parse_circuit(header_match, extra_lines: list[str]) -> Circuit:
    a, b, rest = header_match.groups()
    exprs = [rest] if rest.strip() else []
    exprs += [ln.strip() for ln in extra_lines if ln.strip()]
    tree = _flatten(Series([parse_expr(e) for e in exprs])) if exprs else Series([])
    if not isinstance(tree, Series):
        tree = Series([tree])
    ground = False
    parts = []
    for p in tree.parts:
        if isinstance(p, Comp) and p.kind == "terra":
            ground = True
        else:
            parts.append(p)
    tree.parts = parts
    source = None
    if not a:
        for k, p in enumerate(parts):
            if isinstance(p, Comp) and p.kind in SOURCES:
                source = parts.pop(k)
                break
    terminals = (a or "A", b or "B") if source is None else ()
    return Circuit(tree, source, ground, terminals)


def label_parts(c: Comp) -> tuple[str, str]:
    """(nome, valor) do rótulo; nome vazio quando o desenho já diz o que é."""
    name = c.name or ""
    if _GENERIC.match(name) or (len(name) == 1 and name.isalpha()):
        name = ""
    return name, c.value


def label_text(c: Comp) -> str:
    name = c.name or ""
    # "bateria", "lâmpada" ou uma letra só (V, A, R) já são ditos pelo desenho
    if _GENERIC.match(name) or (len(name) == 1 and name.isalpha()):
        name = ""
    return " ".join(x for x in (name, c.value) if x)


def comps(node) -> list[Comp]:
    if isinstance(node, Comp):
        return [node]
    if isinstance(node, Series):
        return [c for p in node.parts for c in comps(p)]
    if isinstance(node, Parallel):
        return [c for b in node.branches for c in comps(b)]
    if isinstance(node, Circuit):
        return comps(node.tree) + ([node.source] if node.source else [])
    return []


# ---------------------------------------------------------------- desenho
@dataclass
class Block:
    strokes: list
    w: float
    top: float      # relativo à linha dos terminais (y = 0), y para baixo
    bottom: float


def _bounds(strokes) -> tuple[float, float]:
    if not strokes:
        return 0.0, 0.0
    ys = np.concatenate([s[:, 1] for s in strokes])
    return float(ys.min()), float(ys.max())


def _moved(strokes, dx, dy):
    return [s + [dx, dy] for s in strokes]


def _dot(x, y, r):
    t = np.linspace(0, 2 * np.pi, 10)
    pts = []
    for rr in (r, r * 0.6, r * 0.25):   # bolinha "pintada" com caneta
        pts.append(np.stack([x + rr * np.cos(t), y + rr * np.sin(t)], 1))
    return np.vstack(pts)


class Drawer:
    """Desenha a árvore. label_fn(peças) -> Box do rótulo (ou None)."""

    def __init__(self, xh: float, rng: np.random.Generator, label_fn, max_w: float | None = None):
        self.max_w = max_w
        self.xh = xh
        self.U = 1.9 * xh                  # tamanho dos componentes
        self.rng = rng
        self.label_fn = label_fn
        self.wire_gap = 0.55 * self.U

    def wire(self, p0, p1):
        return hand_line(p0, p1, self.xh * 0.45, self.rng)

    def comp(self, c: Comp, vertical: bool = False, mirror: bool = False) -> Block:
        shp = component(c.kind) or component("impedância")
        st, w = shp
        st = [(s - [0.0, AXIS]) * self.U for s in pen(st, self.rng, amp=0.012)]
        w *= self.U
        if mirror:   # só o símbolo vira (o rótulo continua legível)
            st = [np.stack([w - s[:, 0], s[:, 1]], 1) for s in st]
        label = self.label_fn(c.label) if c.label else None
        if vertical:
            # y' = x (de cima para baixo), x' = -y : o "+" da bateria fica em cima
            st = [np.stack([-s[:, 1], s[:, 0]], 1) for s in st]
            strokes = st
            if label is not None and label.strokes:
                lx = min(float(np.concatenate([s[:, 0] for s in st]).min()), -0.4 * self.U) - 0.35 * self.xh - label.w
                ly = w / 2 - (label.top + label.bottom) / 2
                strokes = strokes + [s + [lx, ly] for s in label.strokes]
            return Block(strokes, w, *_bounds(strokes))
        top, bottom = _bounds(st)
        # rótulo mais largo que o símbolo: o componente "estica" com fio dos dois lados
        need = (label.w + 0.6 * self.xh) if (label is not None and label.strokes) else 0.0
        pad = max(0.0, (need - w) / 2)
        strokes = [s + [pad, 0.0] for s in st]
        if pad:
            strokes.append(self.wire((0.0, 0.0), (pad + 0.02 * self.U, 0.0)))
            strokes.append(self.wire((pad + w - 0.02 * self.U, 0.0), (w + 2 * pad, 0.0)))
        W = w + 2 * pad
        if label is not None and label.strokes:
            lx = (W - label.w) / 2
            ly = min(top, -0.3 * self.U) - 0.3 * self.xh - label.bottom
            strokes += [s + [lx, ly] for s in label.strokes]
        top, bottom = _bounds(strokes)
        return Block(strokes, W, min(top, 0.0), max(bottom, 0.0))

    def series(self, node: Series, mirror: bool = False) -> Block:
        strokes, x = [], 0.0
        parts = node.parts[::-1] if mirror else node.parts
        blocks = [self.block(p, mirror) for p in parts]
        for k, b in enumerate(blocks):
            if k:
                strokes.append(self.wire((x, 0.0), (x + self.wire_gap, 0.0)))
                x += self.wire_gap
            strokes += _moved(b.strokes, x, 0.0)
            x += b.w
        top, bottom = _bounds(strokes)
        return Block(strokes, x, min(top, 0.0), max(bottom, 0.0))

    def parallel(self, node: Parallel, mirror: bool = False) -> Block:
        blocks = [self.block(b, mirror) for b in node.branches]
        bus = 0.8 * self.U
        inner = max(b.w for b in blocks)
        W = inner + 2 * bus
        gap = 0.75 * self.U
        ys, y = [], 0.0
        for k, b in enumerate(blocks):
            if k:
                y += blocks[k - 1].bottom + gap - b.top
            ys.append(y)
        strokes = []
        n = len(blocks)
        for k, (b, yk) in enumerate(zip(blocks, ys)):
            x0 = bus + (inner - b.w) / 2
            strokes.append(self.wire((0.0, yk), (x0, yk)))
            strokes += _moved(b.strokes, x0, yk)
            strokes.append(self.wire((x0 + b.w, yk), (W, yk)))
            if k < n - 1:   # nó: três fios se encontram
                strokes.append(_dot(0.0, yk, 0.09 * self.U))
                strokes.append(_dot(W, yk, 0.09 * self.U))
        strokes.append(self.wire((0.0, ys[0]), (0.0, ys[-1])))
        strokes.append(self.wire((W, ys[0]), (W, ys[-1])))
        top, bottom = _bounds(strokes)
        return Block(strokes, W, min(top, 0.0), max(bottom, 0.0))

    def block(self, node, mirror: bool = False) -> Block:
        if isinstance(node, Comp):
            return self.comp(node, mirror=mirror)
        if isinstance(node, Series):
            if not node.parts:
                return Block([self.wire((0.0, 0.0), (1.5 * self.U, 0.0))], 1.5 * self.U, 0.0, 0.0)
            return self.series(node, mirror)
        return self.parallel(node, mirror)

    def ground_at(self, x, y) -> list:
        st, _ = component("terra")
        # o desenho do terra tem a haste em x=0.6, de AXIS-0.9 até AXIS+0.1
        st = [(s - [0.6, AXIS - 0.9]) * self.U * 0.8 + [x, y] for s in pen(st, self.rng, amp=0.012)]
        return st

    def draw(self, circ: Circuit) -> Block:
        U = self.U
        body = self.block(circ.tree)
        strokes = []
        if circ.source is None:
            # terminais em aberto: o--[ ... ]--o
            r = 0.13 * U
            lead = 0.8 * U
            x = r * 2
            strokes.append(np.stack([r * np.cos(np.linspace(0, 2 * np.pi, 14)),
                                     r * np.sin(np.linspace(0, 2 * np.pi, 14))], 1))
            strokes.append(self.wire((r, 0.0), (x + lead, 0.0)))
            strokes += _moved(body.strokes, x + lead, 0.0)
            xe = x + lead + body.w
            strokes.append(self.wire((xe, 0.0), (xe + lead, 0.0)))
            cx = xe + lead + r
            strokes.append(np.stack([cx + r * np.cos(np.linspace(0, 2 * np.pi, 14)),
                                     r * np.sin(np.linspace(0, 2 * np.pi, 14))], 1))
            items = circ.term_items or [[], []]
            for its, px, side in ((items[0], 0.0, -1), (items[1], cx, 1)):
                lb = self.label_fn(its) if its else None
                if lb is not None and lb.strokes:
                    lx = px - lb.w - 0.35 * self.xh if side < 0 else px + r + 0.35 * self.xh
                    strokes += [s + [lx, -(lb.top + lb.bottom) / 2] for s in lb.strokes]
            if circ.ground:
                strokes += self.ground_at(cx, r)
            top, bottom = _bounds(strokes)
            left = min(float(np.concatenate([s[:, 0] for s in strokes]).min()), 0.0)
            strokes = _moved(strokes, -left, 0.0)
            right = float(np.concatenate([s[:, 0] for s in strokes]).max())
            return Block(strokes, right, top, bottom)

        # malha fechada: fonte em pé à esquerda, resto em cima, fio de volta embaixo
        src = self.comp(circ.source, vertical=True)
        st_len = (component(circ.source.kind) or component("bateria"))[1] * U
        side = 1.0 * U
        parts = circ.tree.parts if isinstance(circ.tree, Series) else [circ.tree]
        bottom_blk = None
        if self.max_w and body.w + 2 * side + 3 * U > self.max_w and len(parts) >= 2:
            # não cabe numa fileira: metade vai no fio de baixo, voltando
            widths = [self.block(p).w for p in parts]
            half, acc, k = sum(widths) / 2, 0.0, 0
            while k < len(parts) - 1 and acc + widths[k] <= half:
                acc += widths[k]
                k += 1
            k = max(k, 1)
            body = self.block(Series(parts[:k]))
            bottom_blk = self.block(Series(parts[k:]), mirror=True)
        inner_w = max(body.w, bottom_blk.w if bottom_blk else 0.0)
        if bottom_blk is not None:
            H = max(body.bottom - bottom_blk.top + 1.0 * U, st_len + 1.6 * U)
        else:
            H = max(body.bottom + 1.1 * U, st_len + 1.6 * U)
        xr = side + inner_w + side
        x_top = side + (inner_w - body.w) / 2
        strokes.append(self.wire((0.0, 0.0), (x_top, 0.0)))
        strokes += _moved(body.strokes, x_top, 0.0)
        strokes.append(self.wire((x_top + body.w, 0.0), (xr, 0.0)))
        strokes.append(self.wire((xr, 0.0), (xr, H)))
        if bottom_blk is None:
            strokes.append(self.wire((xr, H), (0.0, H)))
        else:
            xb = side + (inner_w - bottom_blk.w) / 2
            strokes.append(self.wire((xr, H), (xb + bottom_blk.w, H)))
            strokes += _moved(bottom_blk.strokes, xb, H)
            strokes.append(self.wire((xb, H), (0.0, H)))
        ys0 = (H - st_len) / 2
        strokes.append(self.wire((0.0, 0.0), (0.0, ys0)))
        strokes += _moved(src.strokes, 0.0, ys0)
        strokes.append(self.wire((0.0, ys0 + st_len), (0.0, H)))
        if circ.ground:
            strokes.append(_dot(xr * 0.5, H, 0.09 * U))
            strokes.append(self.wire((xr * 0.5, H), (xr * 0.5, H + 0.35 * U)))
            strokes += self.ground_at(xr * 0.5, H + 0.35 * U)
        top, bottom = _bounds(strokes)
        left = min(float(np.concatenate([s[:, 0] for s in strokes]).min()), 0.0)
        strokes = _moved(strokes, -left, 0.0)
        right = float(np.concatenate([s[:, 0] for s in strokes]).max())
        return Block(strokes, right, top, bottom)
