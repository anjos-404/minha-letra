"""Montagem da folha: texto, sinais, frações, potências e tabelas.

O modelo só sabe escrever linhas de texto comum. Este módulo:
  1. lê o texto e separa em peças: trechos de texto, sinais (+ = / ...),
     frações (5/4, (x+1)/2), potências (x^2, x²) e tabelas;
  2. manda só os trechos de texto para o modelo, todos num lote;
  3. monta as peças lado a lado, empilha frações, eleva potências e desenha
     sinais, traços de fração e grades de tabela com traço de caneta.

Tudo é medido na letra da pessoa: o tamanho do "x" (xh) e a largura média de
uma letra vêm da amostra, então sinais e tabelas ficam proporcionais a ela.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .model import HandwritingModel
from .strokes import (ACCENT_LAG, _diacritic_paths, _path_d, align, denoise, offsets_to_coords,
                      split_strokes, to_model_text)
from . import circuit as circ_mod
from .natural import humanize, stroke_svg
from .symbols import AXIS, COMPONENT_NAMES, LETTERLIKE, SUPPORTED, hand_line, pen, shape

SCALE = 2.0          # mesma escala de line_coords
LINE_HEIGHT = 64.0
MARGIN_LEFT = 70.0
MARGIN_TOP = 80.0
PAGE_WIDTH = 900.0


# ======================================================================
# 1. Plano: o que vai em cada linha (sem geometria)
# ======================================================================
@dataclass
class Text:
    text: str
    base: str = ""
    marks: list = field(default_factory=list)
    box: "Box | None" = None
    legibility: float | None = None


@dataclass
class Sym:
    ch: str


@dataclass
class Space:
    pass


@dataclass
class Frac:
    num: list
    den: list


@dataclass
class Sup:
    items: list


@dataclass
class Row:
    kind: str                       # "text", "blank" ou "table"
    items: list = field(default_factory=list)
    slots: int = 1                  # quantas pautas a linha ocupa
    cells: list = field(default_factory=list)
    table: "Table | None" = None
    index_in_table: int = 0
    circuit: object = None          # circ_mod.Circuit quando kind == "circuit"


@dataclass
class Table:
    n_rows: int
    col_units: list
    header: bool = False


@dataclass
class Sub:
    items: list


@dataclass
class Root:
    items: list
    index: list | None = None


@dataclass
class Matrix:
    rows: list                      # linhas -> células -> peças
    delim: str = "["                # "(", "[", "{", "|", "‖" ou "" (sem)
    align: str = "c"                # "c" centralizado, "l" à esquerda (sistemas)
    seps: tuple = ()                # colunas com traço vertical depois (matriz aumentada)


@dataclass
class BigOp:
    op: str                         # "∫", "∮", "∑", "∏" ou "lim"
    lower: list | None = None
    upper: list | None = None
    name: "Text | None" = None      # o "lim" é escrito pelo modelo


SUPERSCRIPTS = {"²": "2", "³": "3", "¹": "1", "⁰": "0", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8",
                "⁹": "9", "ⁿ": "n", "⁺": "+", "⁻": "-", "ᵀ": "T", "ˣ": "x", "ⁱ": "i", "⁽": "(", "⁾": ")"}
SUBSCRIPTS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8",
              "₉": "9", "₊": "+", "₋": "-", "ₐ": "a", "ₑ": "e", "ₒ": "o", "ₓ": "x", "ᵢ": "i", "ⱼ": "j",
              "ₖ": "k", "ₙ": "n", "ₜ": "t", "ₘ": "m", "ₚ": "p", "ₛ": "s", "ₗ": "l", "ₕ": "h", "₍": "(", "₎": ")"}
NORMALIZE = {"−": "-", "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'", "…": "...", " ": " ",
             "∆": "Δ", "Ω": "Ω", "µ": "μ", "ϵ": "ε", "ϑ": "θ", "ϕ": "φ", "⋅": "·",
             "∙": "·", "≦": "≤", "≧": "≥", "⟶": "→", "⟹": "⇒", "⟺": "⇔", "′": "'"}
LIT_LBRACE, LIT_RBRACE = "", ""   # chaves literais (\{ \}) x chaves de agrupamento
ROWBREAK = "\ue002"                             # \\ do LaTeX (troca de linha em matriz)

LATEX_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε", "eta": "η",
    "theta": "θ", "vartheta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω", "Delta": "Δ",
    "Omega": "Ω", "Sigma": "Σ", "Phi": "Φ", "Theta": "Θ", "Pi": "Π", "Lambda": "Λ", "Gamma": "Γ",
    "Psi": "Ψ", "partial": "∂", "nabla": "∇", "angle": "∠", "perp": "⊥", "parallel": "∥", "in": "∈",
    "therefore": "∴", "Rightarrow": "⇒", "implies": "⇒", "Leftrightarrow": "⇔", "iff": "⇔",
    "uparrow": "↑", "downarrow": "↓", "int": "∫", "oint": "∮", "sum": "∑", "prod": "∏", "cdot": "·",
    "times": "×", "div": "÷", "pm": "±", "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
    "ne": "≠", "approx": "≈", "infty": "∞", "to": "→", "rightarrow": "→", "leftarrow": "←",
    "circ": "°", "degree": "°", "prime": "'", "ldots": "⋯", "cdots": "⋯", "dots": "⋯", "vdots": "⋮", "ddots": "⋱",
    "quad": " ", "qquad": " ", "left": "", "right": "", "big": "", "Big": "", "bigg": "", "Bigg": "",
    "displaystyle": "", "limits": "", "nolimits": "", "sin": "sin", "cos": "cos", "tan": "tan",
    "sen": "sen", "tg": "tg", "ln": "ln", "log": "log", "exp": "exp", "lim": "lim", "max": "max",
    "min": "min", "det": "det", "Omicron": "O", "ell": "l",
}
TEXT_CMDS = {"text", "mathrm", "mathbf", "mathit", "operatorname", "textbf", "textit", "mathsf",
             "boldsymbol", "vec", "hat", "bar", "overline", "mathcal", "mbox"}
GREEK = "αβγδεηθλμνρστφχψωΔΩΣΦΘΠΛΓΨ"
LETTER = set(GREEK) | set("∂∇")

_LET = rf"[A-Za-z{GREEK}π]"
_DLET = rf"[A-Zb-df-np-z{GREEK}]"       # depois do "d" de derivada: sem a, e, o (do/da/de são palavras)
_SCR = r"(?:[_^](?:\{[^{}\n]{1,25}\}|\([^()\n]{1,25}\)|[A-Za-z0-9]+)|[⁰¹²³⁴⁵⁶⁷⁸⁹ⁿ₀₁₂₃₄₅₆₇₈₉ₐₑₒₓᵢⱼₖₙₜₘₚₛₗₕ]+)*"
_FUNC = r"(?:sin|cos|tan|sen|tg|ln|log|exp|sec|cot|[A-Za-z])\([^()\n]{1,30}\)"   # sin(x), f(x), n(n+1)
_ROOT_ATOM = r"√(?:\([^()\n]{1,30}\)|\d+(?:[.,]\d+)?|[A-Za-z])"
_ATOM = (rf"(?:{_FUNC}|{_ROOT_ATOM}|\([^()\n]{{1,40}}\)|\{{[^{{}}\n]{{1,40}}\}}|\d+(?:[.,]\d+)?(?:{_LET}(?![A-Za-z]))?"
         rf"|[d∂](?:[²³]|\^\d)?{_DLET}?(?![A-Za-z])|{_LET}\d*(?![A-Za-z]))")
_OPER = _ATOM + _SCR
_FRAC = re.compile(rf"(?P<a>{_OPER})\s*/\s*(?P<b>{_OPER})")
_UNIT = re.compile(r"^[A-Za-z][²³]?$")


def _strip_paren(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == "(" and s[-1] == ")") or (s[0] == "{" and s[-1] == "}")):
        depth = 0
        for k, ch in enumerate(s):   # só tira se o par de fora for o mesmo: (a)+(b) fica
            depth += ch in "({"
            depth -= ch in ")}"
            if depth == 0 and k < len(s) - 1:
                return s
        return s[1:-1]
    return s


def _kind(ch: str) -> str:
    if ch.isspace() or ch == ROWBREAK:
        return "space"
    if ch in "QXZ":
        return "sym"                 # maiúsculas que o modelo não aprendeu: desenhadas
    if ch in "();":
        return "sym"                 # o modelo se enrola com "2024)." e "2025;": desenhados à parte
    if ch in (LIT_LBRACE, LIT_RBRACE):
        return "sym"
    if to_model_text(ch)[0].strip():
        return "text"
    if ch in SUPPORTED:
        return "sym"
    return "bad"


def _prep(s: str) -> str:
    r"""Normaliza caracteres e LaTeX (\alpha, \cdot, $...$) antes de ler."""
    s = "".join(NORMALIZE.get(c, c) for c in s)
    # delimitadores de matemática LaTeX; $ sozinho (R$ 5) fica
    s = re.sub(r"\$\$(.+?)\$\$", r"\1", s)
    s = re.sub(r"\$([^$\n]*?[\\^_{}][^$\n]*?)\$", r"\1", s)
    s = re.sub(r"\\\((.+?)\\\)", r"\1", s)
    s = re.sub(r"\\\[(.+?)\\\]", r"\1", s)
    s = s.replace("\\\\", ROWBREAK)   # \\ do LaTeX: troca de linha (protege antes do resto)
    s = s.replace(r"\{", LIT_LBRACE).replace(r"\}", LIT_RBRACE).replace(r"\%", "%").replace(r"\$", "$")
    s = re.sub(r"\\[,;:! ]", " ", s)

    def cmd(m):
        name = m.group(1)
        if name not in LATEX_SYMBOLS:
            return m.group(0)
        out = LATEX_SYMBOLS[name]
        # "\Delta V" -> "ΔV": o espaço depois de uma letra grega é só separador do LaTeX
        if m.group(2) and not (len(out) == 1 and out in GREEK):
            out += " "
        return out
    s = re.sub(r"\\([A-Za-z]+)( ?)", cmd, s)
    s = s.replace("^\\circ", "°").replace("^{\\circ}", "°").replace("^∘", "°")
    return s


def _group(s: str, i: int):
    """Lê um argumento: {balanceado}, (balanceado) ou um caractere. -> (texto, novo i)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] in "{(":
        op, cl = s[i], "}" if s[i] == "{" else ")"
        depth, j = 0, i
        while j < len(s):
            depth += s[j] == op
            depth -= s[j] == cl
            if depth == 0:
                inner = s[i + 1:j]
                return inner, j + 1
            j += 1
        return s[i + 1:], len(s)
    return s[i], i + 1


def _script_arg(s: str, i: int):
    """Argumento de ^ ou _: {..}, (..), número, ou letras/gregas (x^10, R_eq, V_{out})."""
    if i < len(s) and s[i] in "{(":
        return _group(s, i)
    j = i
    if j < len(s) and s[j] in "+-":
        j += 1
    if j < len(s) and s[j].isdigit():
        while j < len(s) and (s[j].isdigit() or (s[j] in ".," and j + 1 < len(s) and s[j + 1].isdigit())):
            j += 1
    elif j < len(s) and s[j] in GREEK + "∂∞π":
        j += 1                                  # ^π, _θ: uma letra grega só
    else:
        while j < len(s) and s[j].isascii() and (s[j].isalpha() or s[j] in "'"):
            j += 1
    if j == i:
        return None, i
    return s[i:j], j


def _uni_scripts(s: str, i: int, table: dict):
    j = i
    while j < len(s) and s[j] in table:
        j += 1
    return "".join(table[c] for c in s[i:j]), j


def _limits(s: str, i: int, bad: set):
    """Lê _x e ^y (em qualquer ordem, inclusive ₀¹) depois de ∫ ∑ ∏ lim."""
    lower = upper = None
    for _ in range(2):
        if i < len(s) and s[i] == "_" and lower is None:
            arg, i = _script_arg(s, i + 1)
            lower = parse_inline(arg or "", bad, prepped=True)
        elif i < len(s) and s[i] == "^" and upper is None:
            arg, i = _script_arg(s, i + 1)
            upper = parse_inline(arg or "", bad, prepped=True)
        elif i < len(s) and s[i] in SUBSCRIPTS and lower is None:
            arg, i = _uni_scripts(s, i, SUBSCRIPTS)
            lower = parse_inline(arg, bad, prepped=True)
        elif i < len(s) and s[i] in SUPERSCRIPTS and upper is None:
            arg, i = _uni_scripts(s, i, SUPERSCRIPTS)
            upper = parse_inline(arg, bad, prepped=True)
    return lower, upper, i


def _split_top(s: str, seps: str) -> list[str]:
    """Divide s pelos separadores que estão fora de parênteses/colchetes/chaves."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if depth == 0 and ch in seps:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def _matching(s: str, i: int, op: str, cl: str) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == op and (op != cl or j == i):
            depth += 1
        elif s[j] == cl:
            depth -= 1
            if depth == 0:
                return j
    return -1


def _cells(row: str) -> tuple[list[str], list[int]]:
    """Células de uma linha de matriz: vírgula se tiver, senão espaços. '|' marca separador."""
    seps = []
    parts = _split_top(row, "|")
    cells: list[str] = []
    for k, part in enumerate(parts):
        if k:
            seps.append(len(cells) - 1)
        if "," in "".join(_split_top(part, "")) and any(_split_top(part, ",")[1:]):
            cs = [c.strip() for c in _split_top(part, ",")]
        else:
            cs = part.split()
        cells.extend(c for c in cs if c != "")
    return cells, seps


def _matrix_from_rows(rows_txt: list[str], delim: str, bad: set, align: str = "c", whole_rows=False) -> Matrix:
    rows, seps = [], set()
    for r in rows_txt:
        r = r.strip()
        if not r:
            continue
        if whole_rows:
            cells, sp = [r], []
        else:
            cells, sp = _cells(r)
        seps.update(sp)
        rows.append([parse_inline(c, bad, prepped=True) for c in cells])
    n = max((len(r) for r in rows), default=0)
    rows = [r + [[] for _ in range(n - len(r))] for r in rows]
    return Matrix(rows, delim, align, tuple(sorted(seps)))


ENV_DELIM = {"pmatrix": "(", "bmatrix": "[", "Bmatrix": "{", "vmatrix": "|", "Vmatrix": "‖", "matrix": "",
             "smallmatrix": "", "cases": "{", "array": "", "aligned": "", "align": "", "align*": "",
             "dcases": "{", "rcases": "}"}


def _latex_env(s: str, i: int, bad: set):
    r"""\begin{pmatrix} a & b \\ c & d \end{pmatrix} -> (Matrix, novo i)."""
    m = re.match(r"\\begin\{([A-Za-z*]+)\}", s[i:])
    if not m:
        return None, i
    env = m.group(1)
    j = i + m.end()
    spec = ""
    if env == "array" and j < len(s) and s[j] == "{":
        k = _matching(s, j, "{", "}")
        spec, j = s[j + 1:k], k + 1
    end_tag = "\\end{" + env + "}"
    k = s.find(end_tag, j)
    if k < 0:
        return None, i
    body = s[j:k]
    rows_txt = body.split(ROWBREAK)
    rows, seps = [], set()
    for r in rows_txt:
        if not r.strip() or r.strip() == "\\hline":
            continue
        cells = [c.strip() for c in _split_top(r, "&")]
        rows.append([parse_inline(c, bad, prepped=True) for c in cells])
    if spec:
        col = -1
        for ch in spec:
            if ch in "lcr":
                col += 1
            elif ch == "|" and col >= 0:
                seps.add(col)
    n = max((len(r) for r in rows), default=0)
    rows = [r + [[] for _ in range(n - len(r))] for r in rows]
    align = "l" if env in ("cases", "dcases", "aligned", "align", "align*") else "c"
    return Matrix(rows, ENV_DELIM.get(env, ""), align, tuple(sorted(seps))), k + len(end_tag)


def _simple_matrix(s: str, i: int, bad: set):
    """[1 2; 3 4], (a, b; c, d), |1 2; 3 4|, [[1,2],[3,4]], { x+y=3; x-y=1 }."""
    c = s[i]
    if c == "(" and i > 0 and s[i - 1].isalnum():
        return None, i                               # f(x; y) é chamada de função
    cl = {"[": "]", "(": ")", "{": "}", "|": "|"}[c]
    j = _matching(s, i, c, cl) if c != "|" else s.find("|", i + 1)
    if c == "|":
        # |a b; c d| : o próximo "|" fecha, mas só se dentro tiver ";"
        j = s.find("|", i + 1)
    if j <= i:
        return None, i
    body = s[i + 1:j]
    if c == "[" and re.match(r"^\s*\[.*\]\s*$", body) and _split_top(body, ",")[0].strip().startswith("["):
        inner = [x.strip() for x in _split_top(body, ",")]
        if all(x.startswith("[") and x.endswith("]") for x in inner) and len(inner) >= 1:
            rows = [",".join(_split_top(x[1:-1], ",")) for x in inner]
            return _matrix_from_rows(rows, "[", bad), j + 1
    rows = _split_top(body, ";")
    if len(rows) < 2:
        return None, i
    if c == "(" and max(len(_cells(r)[0]) for r in rows) < 2:
        return None, i                               # (a; b) sozinho: deixa como texto
    if c == "{":
        # sistema: toda linha tem uma relação (=, <, >...)
        if not all(re.search(r"[=<>≤≥≠≈]", r) for r in rows if r.strip()):
            return None, i
        return _matrix_from_rows(rows, "{", bad, align="l", whole_rows=True), j + 1
    # matriz só se parecer matemática: citações como (Agbo et al., 2025; Silva, 2024)
    # ou listas de palavras ficam como texto
    if _looks_like_prose(body):
        return None, i
    return _matrix_from_rows(rows, c, bad), j + 1


_MATH_WORDS = {"sin", "cos", "tan", "sen", "tg", "ln", "log", "exp", "sec", "cot", "det", "max", "min", "lim"}


def _looks_like_prose(body: str) -> bool:
    """Tem palavra de verdade (3+ letras que não são função matemática) ou 'et al'?"""
    if re.search(r"\bet\s+al\b", body, re.I):
        return True
    for w in re.findall(r"[A-Za-zÀ-ÿ]{3,}", body):
        if w.lower() not in _MATH_WORDS:
            return True
    return False


def _is_minus(s: str, i: int) -> bool:
    """'-' é sinal de menos (e não hífen de palavra, como em e-mail ou bem-vindo)?"""
    k = i - 1
    while k >= 0 and s[k] == " ":
        k -= 1
    prev = s[k] if k >= 0 else ""
    nxt = s[i + 1] if i + 1 < len(s) else ""
    if nxt == " " or not nxt:
        return prev != "" and not prev.isalpha() or (k < i - 1)   # "a - b", "5 -" ; não "palavra-"
    if prev == "" or k < i - 1:
        return nxt.isdigit() or nxt in "j(√" or nxt.isalpha()       # começo ou depois de espaço: -5, -j2, - x
    return not prev.isalpha() and (nxt.isdigit() or nxt in "j(√.")  # ∠-20, (-16, =-5


def parse_inline(s: str, bad: set, prepped: bool = False) -> list:
    """Uma linha de texto -> lista de peças (Text, Sym, Space, Frac, Sup, Sub, Root, BigOp)."""
    if not prepped:
        s = _prep(s)
    items: list = []
    buf: list[str] = []

    def flush():
        if buf:
            items.extend(_text_items("".join(buf), bad))
            buf.clear()

    def sub(t):
        return parse_inline(t, bad, prepped=True)

    i = 0
    while i < len(s):
        c = s[i]
        # comandos LaTeX com argumentos
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            name = m.group(1) if m else s[i + 1:i + 2]
            j = i + 1 + len(name)
            if name in ("frac", "dfrac", "tfrac", "cfrac"):
                a, j = _group(s, j)
                b, j = _group(s, j)
                flush()
                items.append(Frac(sub(a), sub(b)))
                i = j
                continue
            if name == "sqrt":
                idx = None
                if j < len(s) and s[j] == "[":
                    k = s.find("]", j)
                    idx, j = (s[j + 1:k], k + 1) if k > 0 else (None, j)
                a, j = _group(s, j)
                flush()
                items.append(Root(sub(a), sub(idx) if idx else None))
                i = j
                continue
            if name == "begin":
                mat, j2 = _latex_env(s, i, bad)
                if mat is not None:
                    flush()
                    # \left[ ... \right] em volta de um array vira o delimitador dele
                    if not mat.delim and items and isinstance(items[-1], Sym) and items[-1].ch in "[(|":
                        mat.delim = items.pop().ch
                        k2 = j2
                        while k2 < len(s) and s[k2] == " ":
                            k2 += 1
                        if k2 < len(s) and s[k2] in "])|":
                            j2 = k2 + 1
                    items.append(mat)
                    i = j2
                    continue
            if name in TEXT_CMDS:
                a, j = _group(s, j)
                flush()
                items.extend(sub(a))
                i = j
                continue
            if name:
                bad.add("\\" + name)
            i = j
            continue
        if c == "-" and _is_minus(s, i):
            flush()
            items.append(Sym("−"))                    # menos de conta: desenhado na altura do =
            i += 1
            continue
        if c in "tI" and (i == 0 or not s[i - 1].isalpha()) and (i + 1 >= len(s) or not s[i + 1].isalpha()):
            flush()
            items.append(Sym(c))   # I com barrinhas, para não virar 1 ou l (2I, I₃)                    # v(t), 377t, t = 0: o "t" sozinho a IA confunde com 1/I
            i += 1
            continue
        if (c == "j" and i + 1 < len(s) and (s[i + 1].isdigit() or s[i + 1] in "(ωθφ")
                and (i == 0 or not s[i - 1].isalpha())):
            flush()
            items.append(Sym("j"))                    # j20, j0,5, jω: j dos complexos desenhado
            i += 1
            continue
        if c in "ºª":
            flush()
            if i > 0 and s[i - 1].isdigit():          # 2º, 1ª: ordinal (letra pequena em cima)
                items.append(Sup([Text("o" if c == "º" else "a")]))
            else:
                items.append(Sym("°"))
            i += 1
            continue
        if c in SUPERSCRIPTS or c in SUBSCRIPTS:
            table, Kind = (SUPERSCRIPTS, Sup) if c in SUPERSCRIPTS else (SUBSCRIPTS, Sub)
            arg, i = _uni_scripts(s, i, table)
            flush()
            items.append(Kind(sub(arg)))
            continue
        if c in "^_" and (buf and not buf[-1].isspace() or items):
            arg, j = _script_arg(s, i + 1)
            if arg is not None:
                flush()
                items.append((Sup if c == "^" else Sub)(sub(arg)))
                i = j
                continue
        if c in "∫∮∑∏":
            flush()
            lower, upper, i = _limits(s, i + 1, bad)
            items.append(BigOp(c, lower, upper))
            continue
        if (s.startswith("lim", i) and (i == 0 or not s[i - 1].isalpha())
                and i + 3 < len(s) and s[i + 3] == "_"):
            flush()
            lower, upper, i = _limits(s, i + 3, bad)
            items.append(BigOp("lim", lower, None, Text("lim")))
            continue
        if c in "√∛∜":
            arg, j = (_group(s, i + 1) if i + 1 < len(s) and s[i + 1] in "({" else (None, i + 1))
            if arg is None:
                m = re.match(rf"[0-9.,]+|{_LET}+", s[i + 1:])
                arg, j = (m.group(0), i + 1 + len(m.group(0))) if m else ("", i + 1)
            flush()
            items.append(Root(sub(arg), {"∛": [Text("3")], "∜": [Text("4")]}.get(c)))
            i = j
            continue
        if c == "[":
            k = s.find("]", i)
            name = s[i + 1:k].strip().lower() if k > 0 else ""
            if name in COMPONENT_NAMES:
                flush()
                items.append(Sym("comp:" + COMPONENT_NAMES[name]))
                i = k + 1
                continue
        if c in "[({|":
            mat, j = _simple_matrix(s, i, bad)
            if mat is not None and mat.rows:
                flush()
                items.append(mat)
                i = j
                continue
        if c == "," and 0 < i < len(s) - 1 and s[i - 1].isdigit() and s[i + 1].isdigit():
            flush()
            items.append(Sym(","))  # vírgula decimal: o modelo (inglês) escreve mal
            i += 1
            continue
        m = _FRAC.match(s, i)
        if m:
            before = s[i - 1] if i else " "
            after = s[m.end()] if m.end() < len(s) else " "
            a, b = m.group("a"), m.group("b")
            # unidade depois de número (10 m/s, 5 V/m): barra normal
            k = i - 1
            while k >= 0 and s[k] == " ":
                k -= 1
            unit = _UNIT.match(a) and _UNIT.match(b) and k >= 0 and s[k].isdigit()
            # 28/10/2026, km/h, e/ou: barra normal. 5/4, x/2, (x+1)/2, dy/dx: fração em pé.
            if not unit and not (before.isalnum() or before in "/.,") and not (after.isalnum() or after == "/"):
                flush()
                items.append(Frac(sub(_strip_paren(a)), sub(_strip_paren(b))))
                i = m.end()
                continue
        buf.append(c)
        i += 1
    flush()
    return items


def _text_items(s: str, bad: set) -> list:
    out: list = []
    run: list[str] = []

    def flush_run():
        txt = "".join(run)
        run.clear()
        if not txt:
            return
        lead = len(txt) - len(txt.lstrip())
        trail = len(txt) - len(txt.rstrip())
        if lead:
            out.append(Space())
        core = re.sub(r"\s+", " ", txt.strip())
        # pontuação no começo do trecho (logo depois de ")"): desenhada, senão o modelo some com ela
        while core and core[0] in ".,:;!?" and core[0] in ".,:;":
            out.append(Sym(core[0]))
            core = core[1:]
            if core.startswith(" "):
                out.append(Space())
                core = core.lstrip()
        if core and set(core) <= set(".:;"):
            out.extend(Sym(ch) for ch in core)  # pontuação solta: o modelo dobra os pontos
        elif core and len(core) >= 2 and (core[-1] in ":;" or (core[-1] == "." and core[-2].isdigit())):
            # "Circuito:" -> "Circuito" + ":" desenhado (o modelo às vezes repete a pontuação final)
            out.append(Text(core[:-1]))         # "80." -> "80" + ponto desenhado
            out.append(Sym(core[-1]))
        elif core:
            out.append(Text(core))
        if trail and core:
            out.append(Space())

    for c in s:
        k = _kind(c)
        if k in ("text", "space"):
            run.append(c)
        else:
            flush_run()
            if k == "sym":
                out.append(Sym({LIT_LBRACE: "{", LIT_RBRACE: "}"}.get(c, c)))
            else:
                bad.add(c)
    flush_run()
    # junta espaços repetidos
    clean: list = []
    for it in out:
        if isinstance(it, Space) and clean and isinstance(clean[-1], Space):
            continue
        clean.append(it)
    return clean


def units(items: list) -> float:
    """Largura estimada, em 'letras'."""
    w = 0.0
    for it in items or []:
        if isinstance(it, Text):
            w += len(it.text)
        elif isinstance(it, Sym):
            w += (0.5 if it.ch in ",.:;()" else 3.0 if it.ch.startswith("comp:")
                  else 1.0 if it.ch in LETTERLIKE else 1.4)
        elif isinstance(it, Space):
            w += 1.0
        elif isinstance(it, Frac):
            w += max(units(it.num), units(it.den)) * 0.85 + 0.8
        elif isinstance(it, (Sup, Sub)):
            w += units(it.items) * 0.62 + 0.2
        elif isinstance(it, Root):
            w += units(it.items) + 1.3
        elif isinstance(it, BigOp):
            w += max(2.0 if it.op == "lim" else 1.4, units(it.lower) * 0.6, units(it.upper) * 0.6) + 0.6
        elif isinstance(it, Matrix):
            cols = list(zip(*it.rows)) if it.rows else []
            w += sum(max(units(c) for c in col) * 0.9 for col in cols) + 1.1 * max(len(cols) - 1, 0) + 1.6
    return w


def is_tall(items: list) -> bool:
    """A linha precisa de duas pautas? (frações, ∫ ∑ ∏, limites embaixo)"""
    for it in items or []:
        if isinstance(it, (Frac, BigOp, Matrix)):
            return True
        if isinstance(it, (Sup, Sub)) and is_tall(it.items):
            return True
        if isinstance(it, Root) and is_tall(it.items):
            return True
    return False


has_frac = is_tall  # nome antigo


def _depth_units(items: list) -> float:
    """Altura estimada da linha, em alturas de "x"."""
    h = 1.6
    for it in items or []:
        if isinstance(it, Frac):
            h = max(h, 1.0 + _depth_units(it.num) * 0.85 + _depth_units(it.den) * 0.85)
        elif isinstance(it, BigOp):
            h = max(h, 4.2 if (it.lower or it.upper) else 3.0)
        elif isinstance(it, Matrix):
            rh = [max((_depth_units(c) for c in r), default=1.6) * 0.92 for r in it.rows]
            h = max(h, sum(rh) + 0.85 * max(len(rh) - 1, 0) + 0.8)
        elif isinstance(it, (Sup, Sub, Root)):
            h = max(h, _depth_units(it.items) + 0.6)
    return h


XH_EST = 26.0  # px; só para planejar quantas pautas cada linha ocupa


def row_slots(items: list) -> int:
    if not is_tall(items):
        return 1
    px = _depth_units(items) * XH_EST
    return max(2, int(np.ceil((px + 0.35 * LINE_HEIGHT) / LINE_HEIGHT)))


def wrap(items: list, cap: float) -> list[list]:
    """Quebra em linhas sem separar uma expressão (x+5/2 fica junto)."""
    tokens: list = []
    for it in items:
        if isinstance(it, Text):
            words = it.text.split(" ")
            for k, wd in enumerate(words):
                if k:
                    tokens.append(Space())
                while len(wd) > cap:  # palavra maior que a linha
                    tokens.append(Text(wd[:int(cap) - 1] + "-"))
                    tokens.append(Space())
                    wd = wd[int(cap) - 1:]
                if wd:
                    tokens.append(Text(wd))
        else:
            tokens.append(it)
    groups: list[list] = [[]]
    for t in tokens:
        if isinstance(t, Space):
            if groups[-1]:
                groups.append([])
        else:
            groups[-1].append(t)
    groups = [g for g in groups if g]
    # não quebra a linha no meio de uma conta: "C = 47 μF" fica junto
    REL = set("=≠≈≤≥<>+×÷·→⇒⇔∈±")
    glued: list[list] = []
    for g in groups:
        starts_rel = isinstance(g[0], Sym) and g[0].ch in REL
        ends_rel = glued and isinstance(glued[-1][-1], Sym) and glued[-1][-1].ch in REL
        if glued and (starts_rel or ends_rel) and units(glued[-1]) + units(g) + 1 <= cap:
            glued[-1] = glued[-1] + [Space()] + g
        else:
            glued.append(g)
    groups = glued

    rows: list[list] = []
    cur: list = []
    used = 0.0
    for g in groups:
        gw = units(g)
        if cur and used + 1 + gw > cap:
            rows.append(cur)
            cur, used = [], 0.0
        if cur:
            cur.append(Space())
            used += 1
        cur.extend(g)
        used += gw
    if cur:
        rows.append(cur)
    return [_merge_words(r) for r in rows]


_NUMBER = re.compile(r"^\d[\d.,]*$")


def _merge_words(row: list) -> list:
    """Palavras seguidas viram um trecho só (o modelo escreve com mais fluidez).
    Números ficam separados: sozinhos, ganham mais tentativas e saem mais legíveis."""
    out: list = []
    for it in row:
        if (isinstance(it, Text) and len(out) >= 2 and isinstance(out[-1], Space)
                and isinstance(out[-2], Text)
                and not _NUMBER.match(it.text) and not _NUMBER.match(out[-2].text.split(" ")[-1])):
            out.pop()
            out[-1] = Text(out[-1].text + " " + it.text)
        else:
            out.append(it)
    return out


_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def _table_cells(line: str):
    if "\t" in line:
        cells = line.split("\t")
        return [c.strip() for c in cells] if len(cells) >= 2 else None
    st = line.strip()
    if st.count("|") >= 2 or (st.startswith("|") and st.count("|") >= 1):
        cells = st.strip("|").split("|")
        return [c.strip() for c in cells] if len(cells) >= 2 else None
    return None


def _try_table(lines: list[str], i: int, bad: set):
    rows, header, j = [], False, i
    while j < len(lines):
        if _SEP.match(lines[j]) and "-" in lines[j]:
            if len(rows) == 1:
                header = True
            j += 1
            continue
        cells = _table_cells(lines[j])
        if cells is None:
            break
        rows.append(cells)
        j += 1
    if len(rows) < 2:
        return None, i
    ncol = max(len(r) for r in rows)
    parsed = [[parse_inline(c, bad) for c in r + [""] * (ncol - len(r))] for r in rows]
    col_units = [max(max(units(cell) for cell in col), 1.0) + 1.6 for col in zip(*parsed)]
    table = Table(n_rows=len(parsed), col_units=col_units, header=header)
    out = [Row("table", cells=r, table=table, index_in_table=k) for k, r in enumerate(parsed)]
    return out, j


def plan_document(text: str, cap: float = 40, xh: float | None = None) -> tuple[list[Row], set]:
    """Texto -> linhas planejadas + caracteres que não dá para escrever.

    xh: altura do "x" da letra da pessoa (px), para reservar as pautas certas
    para matrizes e circuitos. Sem ela, usa uma estimativa.
    """
    global XH_EST
    if xh:
        XH_EST = float(xh)
    bad: set = set()
    lines = text.replace("\r", "").split("\n")
    rows: list[Row] = []
    i = 0
    while i < len(lines):
        table, j = _try_table(lines, i, bad)
        if table:
            if rows and rows[-1].kind != "blank":
                rows.append(Row("blank"))       # uma pauta livre antes da tabela
            rows.extend(table)
            i = j
            if i < len(lines) and lines[i].strip():
                rows.append(Row("blank"))       # e outra depois
            continue
        hm = circ_mod.HEADER.match(lines[i])
        if hm:
            extra = []
            j = i + 1
            if not hm.group(3).strip():           # "circuito:" sozinho: as linhas de baixo
                while j < len(lines) and lines[j].strip():
                    extra.append(lines[j])
                    j += 1
            i = j
            rows.append(_circuit_row(hm, extra, bad))
            continue
        line = lines[i]
        i += 1
        if not line.strip():
            rows.append(Row("blank"))
            continue
        for r in wrap(parse_inline(line, bad), cap):
            rows.append(Row("text", items=r, slots=row_slots(r)))
    while rows and rows[-1].kind == "blank":
        rows.pop()
    return rows, bad


def _circuit_row(hm, extra: list[str], bad: set) -> Row:
    c = circ_mod.parse_circuit(hm, extra)
    for comp in circ_mod.comps(c):
        name, value = circ_mod.label_parts(comp)
        # nome e valor em trechos separados: o modelo tende a grudar "C1 47" em "C147"
        parts = [parse_inline(x, bad) for x in (name, value) if x]
        comp.label = (parts[0] + ([Space()] + parts[1] if len(parts) > 1 else [])) if parts else []
    if c.terminals:
        c.term_items = [parse_inline(t, bad) for t in c.terminals]

    # quantas pautas: desenha uma vez com rótulos de mentira, só para medir
    def fake_label(items):
        if not items:
            return None
        w = units(items) * XH_EST * 0.9 * 0.75
        return Box([np.array([[0.0, -1.3 * XH_EST * 0.75], [w, 0.0]])], w, -1.3 * XH_EST * 0.75, 0.0)
    blk = circ_mod.Drawer(XH_EST, np.random.default_rng(0), fake_label,
                          max_w=PAGE_WIDTH - MARGIN_LEFT - 40).draw(c)
    k = min(1.0, (PAGE_WIDTH - MARGIN_LEFT - 30) / max(blk.w, 1.0))
    h = (blk.bottom - blk.top) * k
    slots = max(2, int(np.ceil((h + 0.6 * LINE_HEIGHT) / LINE_HEIGHT)))
    return Row("circuit", circuit=c, slots=slots)


def row_texts(row: Row) -> list[Text]:
    """Todos os trechos de texto de uma linha (inclusive dentro de frações e células)."""
    out: list[Text] = []

    def walk(items):
        for it in items:
            if isinstance(it, Text):
                out.append(it)
            elif isinstance(it, Frac):
                walk(it.num)
                walk(it.den)
            elif isinstance(it, (Sup, Sub)):
                walk(it.items)
            elif isinstance(it, Root):
                walk(it.items)
                walk(it.index or [])
            elif isinstance(it, BigOp):
                if it.name is not None:
                    out.append(it.name)
                walk(it.lower or [])
                walk(it.upper or [])
            elif isinstance(it, Matrix):
                for r in it.rows:
                    for cell in r:
                        walk(cell)

    walk(row.items)
    for cell in row.cells:
        walk(cell)
    if row.circuit is not None:
        for comp in circ_mod.comps(row.circuit):
            walk(comp.label)
        for its in row.circuit.term_items:
            walk(its)
    return out


# ======================================================================
# 2. Medidas da letra (vêm da amostra)
# ======================================================================
@dataclass
class Metrics:
    xh: float        # altura do "x", em px
    char_w: float    # largura média de uma letra, em px


def metrics_from_sample(prime: np.ndarray, prime_text: str) -> Metrics:
    c = denoise(offsets_to_coords(prime.astype(np.float64) * [SCALE, SCALE, 1]))
    c[:, :2] = align(c[:, :2])
    y = -c[:, 1]
    base = np.percentile(y, 88)
    # (base - p40) mede ~2/3 da altura real do "x"; o fator calibra com letras reais
    xh = float(np.clip(1.45 * (base - np.percentile(y, 40)), 8, 50))
    width = np.ptp(c[:, 0])
    char_w = float(np.clip(width / max(len(prime_text), 1), 4, 40))
    return Metrics(xh=xh, char_w=char_w)


# ======================================================================
# 3. Geometria
# ======================================================================
@dataclass
class Box:
    strokes: list            # arrays (N,2) em px, y para baixo, linha de base em y=0
    w: float
    top: float               # menor y (negativo = acima da base)
    bottom: float

    def moved(self, dx: float, dy: float) -> "Box":
        return Box([s + [dx, dy] for s in self.strokes], self.w, self.top + dy, self.bottom + dy)

    def scaled(self, k: float) -> "Box":
        return Box([s * k for s in self.strokes], self.w * k, self.top * k, self.bottom * k)


def _bounds(strokes: list) -> tuple[float, float]:
    if not strokes:
        return 0.0, 0.0
    ys = np.concatenate([s[:, 1] for s in strokes])
    return float(ys.min()), float(ys.max())


DESC = set("gjpqy,;")


def text_box(t: Text, rng: np.random.Generator) -> Box:
    off, cidx = t.result
    c = denoise(offsets_to_coords(off.astype(np.float64) * [SCALE, SCALE, 1]))
    if len(t.base.strip()) >= 5:          # tirar a inclinação só faz sentido em trechos longos
        c[:, :2] = align(c[:, :2])
    c[:, 1] *= -1
    y = c[:, 1]
    short = len(t.base.strip()) <= 3
    base = None
    if len(t.base.strip()) <= 6:
        base = _letter_baseline(y, np.asarray(cidx), t.base)
    if base is None:
        if short and not (set(t.base) & DESC):
            base = float(np.percentile(y, 97))
        elif short:
            base = float(np.percentile(y, 70))
        else:
            base = float(np.percentile(y, 88))
    xy = c[:, :2] - [c[:, 0].min(), base]
    strokes = []
    for st in split_strokes(np.hstack([xy, c[:, 2:3]])):
        if len(st) == 1:
            st = np.vstack([st, st + [0.6, 0.6, 0]])
        strokes.append(st[:, :2])
    strokes += _accent_strokes(xy, np.asarray(cidx), t.marks, rng)
    top, bottom = _bounds(strokes)
    return Box(strokes, float(xy[:, 0].max()), top, bottom)


def _letter_baseline(y: np.ndarray, cidx: np.ndarray, text: str):
    """Base de um trecho curto: a base de cada letra que não desce (g, j, p, q, y, vírgula...)
    e a mediana delas. Um rabisco solto ou a perninha do j não puxam a linha."""
    if len(cidx) != len(y) or len(y) < 6:
        return None
    lag = ACCENT_LAG
    lab = np.concatenate([np.full(lag, cidx[0]), cidx[:len(cidx) - lag]]) if len(cidx) > lag else cidx
    bottoms = []
    for k, ch in enumerate(text):
        if ch in DESC or not ch.isalnum():
            continue
        sel = lab == k
        if sel.sum() >= 3:
            bottoms.append(float(np.percentile(y[sel], 92)))
    if not bottoms:
        return None
    return float(np.median(bottoms))


def _accent_strokes(xy, cidx, marks, rng) -> list:
    if not any(marks) or len(cidx) == 0:
        return []
    size = LINE_HEIGHT * 0.55
    lag = ACCENT_LAG
    lab = np.concatenate([np.full(lag, cidx[0]), cidx[:len(cidx) - lag]]) if len(cidx) > lag else cidx
    out = []
    for k, mark in enumerate(marks):
        if not mark:
            continue
        sel = np.where(lab == k)[0]
        if len(sel) < 2:
            sel = np.where(cidx == k)[0]
        if len(sel) < 2:
            continue
        x0, x1 = np.percentile(xy[sel, 0], [10, 90])
        cx = (x0 + x1) / 2
        near = np.abs(xy[:, 0] - cx) <= max((x1 - x0) * 0.35, 2.0)
        pts = xy[near] if near.sum() >= 2 else xy[sel]
        top = float(np.percentile(pts[:, 1], 3))
        bottom = float(np.percentile(xy[sel, 1], 92))
        for d in _diacritic_paths(mark, cx, top, bottom, size, rng):
            nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", d)]
            out.append(np.array(nums).reshape(-1, 2))
    return out


def sym_box(ch: str, xh: float, rng: np.random.Generator) -> Box:
    shp = shape(ch)
    if shp is None:
        return Box([], 0.0, 0.0, 0.0)
    strokes, w = shp
    strokes = [s * xh for s in pen(strokes, rng)]
    top, bottom = _bounds(strokes)
    return Box(strokes, w * xh, top, bottom)


def lay(items: list, m: Metrics, rng: np.random.Generator, scale: float = 1.0) -> Box:
    """Monta uma sequência de peças lado a lado. scale encolhe (frações, expoentes)."""
    xh = m.xh * scale
    strokes, x = [], 0.0
    top, bottom = 0.0, 0.0
    prev = None
    prev_top = -xh
    script_x = script_end = 0.0
    for it in items or []:
        if isinstance(it, Space):
            x += 0.9 * xh
            prev = it
            continue
        gap = 0.0
        if prev is not None and not isinstance(prev, Space):
            gap = 0.15 * xh
            if isinstance(it, Sym) or isinstance(prev, Sym):
                gap = 0.28 * xh
            if (isinstance(it, Sym) and it.ch in LETTERLIKE) or (isinstance(prev, Sym) and prev.ch in LETTERLIKE):
                gap = 0.12 * xh                      # letras gregas, X, Z: espaçamento de letra
            if (isinstance(it, Sym) and it.ch in ",.:;)") or (isinstance(prev, Sym) and prev.ch in ",("):
                gap = 0.06 * xh
        if isinstance(it, Text):
            if it.box is None:
                continue
            b = it.box.scaled(scale)
        elif isinstance(it, Sym):
            b = sym_box(it.ch, xh, rng)
        elif isinstance(it, Frac):
            b = frac_box(it, m, rng, scale)
        elif isinstance(it, (Sup, Sub)):
            inner = lay(it.items, m, rng, scale * 0.62)
            if isinstance(it, Sup):
                # sobe acompanhando a letra de antes (x² mais baixo, V² mais alto)
                raise_by = max(0.95 * xh, -prev_top - 0.3 * xh) if not isinstance(prev, (Sup, Sub)) else 0.95 * xh
                b = inner.moved(0, -raise_by)
            else:
                b = inner.moved(0, 0.32 * xh)
            gap = 0.05 * xh
            # x₁² : índice e expoente no mesmo lugar, um em cima do outro
            if isinstance(prev, (Sup, Sub)) and type(prev) is not type(it):
                x = script_x - gap
        elif isinstance(it, Root):
            b = root_box(it, m, rng, scale)
        elif isinstance(it, BigOp):
            b = bigop_box(it, m, rng, scale)
        elif isinstance(it, Matrix):
            b = matrix_box(it, m, rng, scale)
        else:
            continue
        x += gap
        bb = b.moved(x, 0)
        strokes += bb.strokes
        top, bottom = min(top, bb.top), max(bottom, bb.bottom)
        if isinstance(it, (Sup, Sub)):
            if not (isinstance(prev, (Sup, Sub)) and type(prev) is not type(it)):
                script_x = x
                script_end = x + b.w
            else:
                script_end = max(script_end, x + b.w)
            x = script_end
        else:
            x += b.w
            prev_top = b.top if b.strokes else -xh
        prev = it
    return Box(strokes, x, top, bottom)


def delimiter(ch: str, y0: float, y1: float, xh: float, rng, right: bool = False) -> tuple[list, float]:
    """Parênteses/colchetes/chaves/barras altos, feitos à mão. -> (traços, largura)."""
    h = y1 - y0
    w = min(0.5 * xh, 0.12 * h + 0.25 * xh)
    mid = (y0 + y1) / 2
    if ch == "(":
        pts = [(w, y0), (0.25 * w, y0 + 0.2 * h), (0.0, mid), (0.25 * w, y1 - 0.2 * h), (w, y1)]
        strokes = [_smooth_px(pts)]
    elif ch == "[":
        strokes = [np.vstack([hand_line((w, y0), (0.05 * w, y0), xh * 0.5, rng),
                              hand_line((0.05 * w, y0), (0.05 * w, y1), xh * 0.5, rng),
                              hand_line((0.05 * w, y1), (w, y1), xh * 0.5, rng)])]
    elif ch == "{":
        pts = [(w, y0), (0.45 * w, y0 + 0.05 * h), (0.45 * w, mid - 0.08 * h), (0.0, mid),
               (0.45 * w, mid + 0.08 * h), (0.45 * w, y1 - 0.05 * h), (w, y1)]
        strokes = [_smooth_px(pts, 2)]
    elif ch == "}":
        s_, w = delimiter("{", y0, y1, xh, rng)
        strokes = [np.stack([w - s_[0][:, 0], s_[0][:, 1]], 1)]
        return strokes, w
    elif ch == "|":
        strokes, w = [hand_line((0.3 * xh, y0), (0.3 * xh, y1), xh * 0.5, rng)], 0.45 * xh
    elif ch == "‖":
        strokes = [hand_line((0.2 * xh, y0), (0.2 * xh, y1), xh * 0.5, rng),
                   hand_line((0.5 * xh, y0), (0.5 * xh, y1), xh * 0.5, rng)]
        w = 0.7 * xh
    else:
        return [], 0.0
    if right and ch in "([":
        strokes = [np.stack([w - s_[:, 0], s_[:, 1]], 1) for s_ in strokes]
    return strokes, w


def _smooth_px(pts, rounds: int = 3) -> np.ndarray:
    p = np.array(pts, float)
    for _ in range(rounds):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        p = np.vstack([p[:1], np.column_stack([q, r]).reshape(-1, 2), p[-1:]])
    return p


def matrix_box(mt: Matrix, m: Metrics, rng, scale: float) -> Box:
    """Matriz/sistema: células alinhadas em colunas, delimitadores do tamanho do conteúdo."""
    xh = m.xh * scale
    cs = scale * 0.92
    boxes = [[lay(cell, m, rng, cs) for cell in r] for r in mt.rows]
    if not boxes:
        return Box([], 0.0, 0.0, 0.0)
    ncol = max(len(r) for r in boxes)
    col_w = [max((r[c].w for r in boxes if c < len(r)), default=0.0) for c in range(ncol)]
    col_w = [max(w, 0.6 * xh) for w in col_w]
    asc = [max([-b.top for b in r] + [1.0 * xh * 0.92]) for r in boxes]
    desc = [max([b.bottom for b in r] + [0.15 * xh]) for r in boxes]
    gap_y, gap_x = 0.85 * xh, 1.3 * xh
    # baselines de cada linha, começando em 0
    ys, y = [], 0.0
    for k in range(len(boxes)):
        if k:
            y += desc[k - 1] + gap_y + asc[k]
        ys.append(y)
    top_y = ys[0] - asc[0]
    bot_y = ys[-1] + desc[-1]
    shift = AXIS * xh - (top_y + bot_y) / 2       # centra no eixo (altura do traço do =)
    ys = [v + shift for v in ys]
    top_y += shift
    bot_y += shift
    pad = 0.3 * xh
    left, lw = delimiter(mt.delim, top_y - pad, bot_y + pad, xh, rng) if mt.delim else ([], 0.0)
    x0 = lw + (0.25 * xh if mt.delim else 0.0)
    strokes = list(left)
    xs = []
    x = x0
    for c in range(ncol):
        xs.append(x)
        x += col_w[c] + (gap_x if c < ncol - 1 else 0.0)
    inner_w = x - x0
    for r, row in enumerate(boxes):
        for c, b in enumerate(row):
            if not b.strokes:
                continue
            dx = xs[c] if mt.align == "l" else xs[c] + (col_w[c] - b.w) / 2
            strokes += b.moved(dx, ys[r]).strokes
    for c in mt.seps:
        if c < ncol - 1:
            sx = xs[c] + col_w[c] + gap_x / 2
            strokes.append(hand_line((sx, top_y - pad * 0.5), (sx, bot_y + pad * 0.5), xh * 0.5, rng))
    W = x0 + inner_w + (0.25 * xh if mt.delim else 0.0)
    right_ch = {"(": ")", "[": "]", "|": "|", "‖": "‖", "{": ""}.get(mt.delim, "")
    if right_ch:
        rs, rw = delimiter("(" if right_ch == ")" else "[" if right_ch == "]" else right_ch,
                           top_y - pad, bot_y + pad, xh, rng, right=right_ch in ")]")
        strokes += [s_ + [W, 0.0] for s_ in rs]
        W += rw
    top, bottom = _bounds(strokes)
    return Box(strokes, W + 0.05 * xh, min(top, 0.0), max(bottom, 0.0))


def _xh_units(strokes, xh):
    return [np.asarray(s, float) * xh for s in strokes]


def root_box(r: Root, m: Metrics, rng, scale: float) -> Box:
    """√ com a barra cobrindo o que está dentro (e índice, no ∛)."""
    xh = m.xh * scale
    inner = lay(r.items, m, rng, scale)
    T = min(inner.top, -1.0 * xh) - 0.22 * xh       # altura da barra
    B = max(inner.bottom * 0.4, 0.0) + 0.12 * xh     # ponta de baixo do "check"
    lead = 0.0
    idx = None
    if r.index:
        idx = lay(r.index, m, rng, scale * 0.5)
        lead = max(0.0, idx.w - 0.25 * xh)
    hook_w = 0.85 * xh
    pts = np.array([(lead + 0.0, -0.42 * xh), (lead + 0.16 * xh, -0.52 * xh), (lead + 0.4 * xh, B),
                    (lead + hook_w, T), (lead + hook_w + inner.w + 0.25 * xh, T)])
    radical = []
    for k in range(len(pts) - 1):
        radical.append(hand_line(pts[k], pts[k + 1], xh * 0.6, rng) if k == 3 else
                       np.linspace(pts[k], pts[k + 1], 8))
    path = np.vstack(radical)
    strokes = [path] + inner.moved(lead + hook_w + 0.12 * xh, 0).strokes
    if idx is not None:
        strokes += idx.moved(0, -0.62 * xh - idx.bottom).strokes
    top, bottom = _bounds(strokes)
    return Box(strokes, lead + hook_w + inner.w + 0.3 * xh, top, bottom)


def bigop_box(op: BigOp, m: Metrics, rng, scale: float) -> Box:
    """∫ ∮ ∑ ∏ lim com os limites (em cima/embaixo ou à direita, no ∫)."""
    xh = m.xh * scale
    if op.op == "lim":
        if op.name is None or op.name.box is None:
            return Box([], 0.0, 0.0, 0.0)
        core = op.name.box.scaled(scale)
    else:
        core = sym_box(op.op, xh, rng)
    lower = lay(op.lower, m, rng, scale * 0.6) if op.lower else None
    upper = lay(op.upper, m, rng, scale * 0.6) if op.upper else None
    strokes = []
    if op.op in "∫∮":
        W = core.w
        strokes += core.strokes
        if upper:
            u = upper.moved(core.w * 0.78, core.top + 0.1 * xh - upper.top)
            strokes += u.strokes
            W = max(W, core.w * 0.78 + upper.w)
        if lower:
            lo = lower.moved(core.w * 0.42, core.bottom + 0.05 * xh - lower.bottom)
            strokes += lo.strokes
            W = max(W, core.w * 0.42 + lower.w)
        W += 0.12 * xh
    else:
        W = max(core.w, lower.w if lower else 0.0, upper.w if upper else 0.0)
        strokes += core.moved((W - core.w) / 2, 0).strokes
        if upper:
            strokes += upper.moved((W - upper.w) / 2, core.top - 0.18 * xh - upper.bottom).strokes
        if lower:
            below = max(core.bottom, 0.0 if op.op == "lim" else core.bottom)
            strokes += lower.moved((W - lower.w) / 2, below + 0.2 * xh - lower.top).strokes
    top, bottom = _bounds(strokes)
    return Box(strokes, W, min(top, 0.0), max(bottom, 0.0))


def frac_box(f: Frac, m: Metrics, rng, scale: float) -> Box:
    xh = m.xh * scale
    k = 0.85
    num = lay(f.num, m, rng, scale * k)
    den = lay(f.den, m, rng, scale * k)
    W = max(num.w, den.w) + 0.5 * xh
    bar_y = AXIS * xh
    gap = 0.34 * xh
    n = num.moved((W - num.w) / 2, bar_y - gap - num.bottom)
    d = den.moved((W - den.w) / 2, bar_y + gap - den.top)
    bar = hand_line((0.05 * xh, bar_y), (W - 0.05 * xh, bar_y), xh, rng)
    strokes = n.strokes + d.strokes + [bar]
    top, bottom = _bounds(strokes)
    return Box(strokes, W, min(top, 0), max(bottom, 0))


# ======================================================================
# 4. Página
# ======================================================================
def rule_y(slot: int) -> float:
    """y da pauta onde a escrita do slot se apoia."""
    return MARGIN_TOP + LINE_HEIGHT * slot + LINE_HEIGHT * 0.22


def slot_starts(rows: list[Row]) -> list[int]:
    out, s = [], 0
    for r in rows:
        out.append(s)
        s += r.slots
    return out


def _fit(box: Box, max_w: float, max_h: float | None = None) -> float:
    k = 1.0
    if box.w > max_w:
        k = max_w / box.w
    if max_h is not None and (box.bottom - box.top) * k > max_h:
        k = max_h / max(box.bottom - box.top, 1e-6)
    return k


def row_strokes(row: Row, start_slot: int, m: Metrics, rng, nat: float = 0.0) -> list[np.ndarray]:
    """Traços de uma linha, já na posição da folha (px). nat = naturalidade (0..2)."""
    avail = PAGE_WIDTH - MARGIN_LEFT - 20
    if row.kind == "blank":
        return []
    if row.kind == "text":
        box = lay(row.items, m, rng)
        base = rule_y(start_slot + row.slots - 1)  # apoia na pauta; o numerador usa a de cima
        k = _fit(box, avail)
        if row.slots > 1:
            # Faixa livre da linha: logo abaixo da escrita de cima até antes da de baixo.
            # Cabe apoiada na pauta? ótimo. Se não, sobe o que precisar, e só encolhe
            # se nem assim couber (∑ com limites, fração dentro de fração).
            y_top = rule_y(start_slot) - 0.78 * LINE_HEIGHT
            below = any(isinstance(it, BigOp) and it.lower and it.op in "∑∏lim" for it in row.items)
            # com limite embaixo (∑, lim), tudo sobe e o limite fica acima da pauta
            y_bot = rule_y(start_slot + row.slots - 1) + (0.02 if below else 0.2) * LINE_HEIGHT
            h = (box.bottom - box.top) * k
            if h > y_bot - y_top:
                k *= (y_bot - y_top) / h
            if row.slots >= 3:
                # matriz alta: centraliza na faixa (com respiro em relação à linha de cima)
                y_top = rule_y(start_slot) - 0.66 * LINE_HEIGHT
                h = (box.bottom - box.top) * k
                if h > y_bot - y_top:
                    k *= (y_bot - y_top) / h
                base = (y_top + y_bot) / 2 - (box.top + box.bottom) / 2 * k
            if base + box.bottom * k > y_bot:
                base = y_bot - box.bottom * k
            if base + box.top * k < y_top:
                base = y_top - box.top * k
        out = [s * k + [MARGIN_LEFT, base] for s in box.strokes]
        # linhas altas (fração, matriz) variam menos: o espaço entre as pautas é justo
        return humanize(out, x0=MARGIN_LEFT, base=base, xh=m.xh, amount=nat * (0.6 if row.slots > 1 else 1.0),
                        rng=rng, max_x=PAGE_WIDTH - 12)
    if row.kind == "circuit":
        def label_fn(items):
            return lay(items, m, rng, 0.75) if items else None
        blk = circ_mod.Drawer(m.xh, rng, label_fn, max_w=avail - 20).draw(row.circuit)
        y_top = rule_y(start_slot) - 0.62 * LINE_HEIGHT
        y_bot = rule_y(start_slot + row.slots - 1) - 0.05 * LINE_HEIGHT
        k = min(1.0, (avail - 10) / max(blk.w, 1.0), (y_bot - y_top) / max(blk.bottom - blk.top, 1.0))
        x0 = MARGIN_LEFT + 0.5 * m.xh
        y0 = (y_top + y_bot) / 2 - (blk.top + blk.bottom) / 2 * k
        return [s * k + [x0, y0] for s in blk.strokes]
    # tabela: uma linha da tabela por pauta
    xs = table_columns(row.table, m)
    top = rule_y(start_slot) - LINE_HEIGHT
    bottom = rule_y(start_slot)
    out = []
    for c, cell in enumerate(row.cells):
        if not cell:
            continue
        box = lay(cell, m, rng)
        if not box.strokes:
            continue
        pad = 0.35 * m.xh
        k = _fit(box, xs[c + 1] - xs[c] - 2 * pad, LINE_HEIGHT * 0.78)
        cy = (top + bottom) / 2 + 0.06 * LINE_HEIGHT
        dy = cy - (box.top + box.bottom) / 2 * k
        cell = [s * k + [xs[c] + pad, dy] for s in box.strokes]
        # dentro da tabela: só a base ondulando e a inclinação, sem mexer no começo
        out += humanize(cell, x0=xs[c] + pad, base=dy, xh=m.xh, amount=nat * 0.6, rng=rng,
                        max_x=xs[c + 1] - 3, line_level=False)
    return out


def table_columns(t: Table, m: Metrics) -> list[float]:
    avail = PAGE_WIDTH - MARGIN_LEFT - 30
    widths = np.array(t.col_units) * m.char_w * 1.05
    if widths.sum() > avail:
        widths *= avail / widths.sum()
    x0 = MARGIN_LEFT - 8
    return [x0] + list(x0 + np.cumsum(widths))


def table_grid(t: Table, start_slot: int, m: Metrics, rng) -> list[np.ndarray]:
    xs = table_columns(t, m)
    y0 = rule_y(start_slot) - LINE_HEIGHT
    ys = [y0 + LINE_HEIGHT * k for k in range(t.n_rows + 1)]
    lines = []
    over = 0.15 * m.xh
    for k, y in enumerate(ys):
        lines.append(hand_line((xs[0] - over, y), (xs[-1] + over, y), m.xh, rng))
        if t.header and k == 1:  # cabeçalho: traço duplo
            lines.append(hand_line((xs[0], y + 3), (xs[-1], y + 3), m.xh, rng))
    for x in xs:
        lines.append(hand_line((x, ys[0] - over), (x, ys[-1] + over), m.xh, rng))
    return lines


BATCH_SEQS = 384   # sequências por lote (trechos x tentativas)


def generate_rows(model: HandwritingModel, rows: list[Row], *, prime, prime_text, bias, seed, tries=4, progress=None):
    """Escreve (com o modelo) todos os trechos de texto das linhas dadas, num lote."""
    texts = [t for r in rows for t in row_texts(r)]
    for t in texts:
        t.base, t.marks = to_model_text(t.text)
    todo = [t for t in texts if t.base.strip()]
    if not todo:
        return
    # trechos curtos (números, siglas) erram mais e custam pouco: mais tentativas.
    # Textos longos vão em lotes de tamanho parecido: o lote anda no ritmo do maior
    # trecho dele, e lotes limitados não estouram a memória.
    for group, n in (([t for t in todo if len(t.base.strip()) > 4], tries),
                     ([t for t in todo if len(t.base.strip()) <= 4], tries * 2)):
        group = sorted(group, key=lambda t: len(t.base))
        per_batch = max(1, BATCH_SEQS // n)
        for k in range(0, len(group), per_batch):
            chunk = group[k:k + per_batch]
            outs = model.generate_best([t.base for t in chunk], tries=n, bias=bias, prime_strokes=prime,
                                       prime_text=prime_text, seed=None if seed is None else seed + k)
            for t, res, lg in zip(chunk, outs, model.last_legibility):
                t.result = res
                t.legibility = lg
            if progress:
                progress(min(k + per_batch, len(group)), len(group))
    rng = np.random.default_rng(seed)
    for t in todo:
        t.box = text_box(t, rng)


def weak_rows(rows: list[Row]) -> list[int]:
    out = []
    for i, r in enumerate(rows):
        if any(t.legibility is not None and t.legibility < 0.45 for t in row_texts(r)):
            out.append(i + 1)
    return out


def _d(s: np.ndarray) -> str:
    return _path_d(s)


PAGE_SLOTS = 18   # pautas por folha: com 40 letras por linha, 18 pautas dão a proporção de uma A4


def paginate(rows: list[Row], page_slots: int = PAGE_SLOTS) -> list[tuple[int, int]]:
    """(página, pauta inicial dentro da página) de cada linha.

    Não corta tabela nem linha alta no meio; linha em branco no topo da folha é pulada.
    """
    out: list[tuple[int, int]] = []
    page, used = 0, 0
    i = 0
    while i < len(rows):
        r = rows[i]
        need = r.slots
        if r.kind == "table" and r.index_in_table == 0:
            need = sum(x.slots for x in rows[i:i + r.table.n_rows])   # a tabela inteira junto
        if r.kind == "blank" and used == 0 and page > 0:
            out.append((page, 0))                    # em branco no topo: não ocupa
            i += 1
            continue
        if used and used + min(need, page_slots) > page_slots:
            page, used = page + 1, 0
            if r.kind == "blank":
                out.append((page, 0))
                i += 1
                continue
        out.append((page, used))
        used += r.slots
        i += 1
    return out


def render_pages(rows: list[Row], m: Metrics, *, page_slots: int = PAGE_SLOTS, **kw) -> list[str]:
    """Uma folha (SVG) por página."""
    pos = paginate(rows, page_slots)
    n_pages = max((p for p, _ in pos), default=0) + 1
    pages = []
    for pg in range(n_pages):
        entries = [(i, r, st) for i, (r, (p, st)) in enumerate(zip(rows, pos)) if p == pg]
        if pg > 0 and all(r.kind == "blank" for _, r, _ in entries):
            continue
        need = max((st + r.slots for _, r, st in entries), default=0)
        pages.append(_render_svg(entries, m, n_slots=max(page_slots, need), **kw))
    return pages


def render_document(rows: list[Row], m: Metrics, *, paper="pautado", ink="#1d2f7a", stroke_width=2.2,
                    seed=None, min_slots=14, naturalness: float = 1.0) -> str:
    """Tudo numa folha só (sem páginas)."""
    entries = list(zip(range(len(rows)), rows, slot_starts(rows)))
    total = sum(r.slots for r in rows)
    return _render_svg(entries, m, n_slots=max(total, min_slots), paper=paper, ink=ink, stroke_width=stroke_width,
                       seed=seed, naturalness=naturalness)


def _render_svg(entries, m: Metrics, *, n_slots: int, paper="pautado", ink="#1d2f7a", stroke_width=2.2,
                seed=None, naturalness: float = 1.0) -> str:
    rng = np.random.default_rng(None if seed is None else seed + (entries[0][0] if entries else 0))
    height = MARGIN_TOP + LINE_HEIGHT * (n_slots + 1)
    width = PAGE_WIDTH
    bg = []
    if paper == "pautado":
        for i in range(n_slots + 1):
            yy = rule_y(i)
            bg.append(f'<line x1="0" y1="{yy:.1f}" x2="{width}" y2="{yy:.1f}" stroke="#9cc3e6" stroke-width="1"/>')
        bg.append(f'<line x1="{MARGIN_LEFT - 16}" y1="0" x2="{MARGIN_LEFT - 16}" y2="{height}" stroke="#e9a3a3" stroke-width="1.2"/>')
    elif paper == "quadriculado":
        g = LINE_HEIGHT / 3
        for i in range(int(width / g) + 1):
            bg.append(f'<line x1="{i * g:.1f}" y1="0" x2="{i * g:.1f}" y2="{height}" stroke="#d6e4f0" stroke-width=".8"/>')
        for i in range(int(height / g) + 1):
            bg.append(f'<line x1="0" y1="{i * g:.1f}" x2="{width}" y2="{i * g:.1f}" stroke="#d6e4f0" stroke-width=".8"/>')

    groups = []
    seen_tables = set()
    for i, row, st in entries:
        if row.kind == "table" and id(row.table) not in seen_tables:
            seen_tables.add(id(row.table))
            grid = table_grid(row.table, st, m, rng)
            groups.append('<g class="grade">' + stroke_svg(grid, base_w=stroke_width * 0.85,
                                                             amount=naturalness * 0.5, rng=rng) + "</g>")
        if row.kind == "blank":
            continue
        groups.append(row_group(i, row, st, m, rng, nat=naturalness, stroke_width=stroke_width))
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height:.0f}" width="{width}" height="{height:.0f}">'
        f'<g class="fundo"><rect width="100%" height="100%" fill="#fffdf7"/>{"".join(bg)}</g>'
        f'<g class="tinta" fill="none" stroke="{ink}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round">'
        f'{"".join(groups)}</g></svg>'
    )


def row_body(row: Row, start_slot: int, m: Metrics, rng, nat: float = 1.0, stroke_width: float = 2.2) -> str:
    """SVG dos traços de uma linha (sem a área de toque)."""
    st = row_strokes(row, start_slot, m, rng, nat=nat)
    # circuitos: sem a variação de linha (fios retos ficam melhores), só a pressão
    return stroke_svg(st, base_w=stroke_width, amount=nat * (0.6 if row.kind == "circuit" else 1.0), rng=rng)


def row_group(i: int, row: Row, start_slot: int, m: Metrics, rng, nat: float = 1.0, stroke_width: float = 2.2) -> str:
    top = rule_y(start_slot) - LINE_HEIGHT * (1.0 if row.kind == "table" else 0.7 if row.kind == "circuit" else 0.8)
    h = LINE_HEIGHT * row.slots
    hit = f'<rect class="hit" x="0" y="{top:.1f}" width="{PAGE_WIDTH}" height="{h:.1f}" fill="transparent" stroke="none"/>'
    return f'<g class="linha" data-line="{i}">{hit}{row_body(row, start_slot, m, rng, nat, stroke_width)}</g>'
