"""Leitura do texto: o que vira letra (IA), símbolo desenhado, fração, matriz..."""
import pytest

from hand.layout import (BigOp, Frac, Matrix, Root, Space, Sub, Sup, Sym, Text, parse_inline, plan_document,
                         row_texts)


def flat(items):
    """Representação compacta: 'texto' | <símbolo> | _ | Frac(...) ..."""
    out = []
    for x in items:
        if isinstance(x, Text):
            out.append(repr(x.text))
        elif isinstance(x, Sym):
            out.append(f"<{x.ch}>")
        elif isinstance(x, Space):
            out.append("_")
        elif isinstance(x, Frac):
            out.append(f"Frac({' '.join(flat(x.num))} / {' '.join(flat(x.den))})")
        elif isinstance(x, Sup):
            out.append(f"Sup({' '.join(flat(x.items))})")
        elif isinstance(x, Sub):
            out.append(f"Sub({' '.join(flat(x.items))})")
        elif isinstance(x, Root):
            out.append(f"Root({' '.join(flat(x.items))})")
        else:
            out.append(type(x).__name__)
    return out


def P(s):
    bad = set()
    items = parse_inline(s, bad)
    return flat(items), bad


def all_text(items):
    from hand.layout import Row
    return "".join(t.text for t in row_texts(Row("text", items=items)))


# ---------------------------------------------------------------- variável t
@pytest.mark.parametrize("s", ["v(t)", "377t", "em t = 0", "cos(t + 30°)", "teste t", "t"])
def test_t_isolado_vira_simbolo(s):
    f, _ = P(s)
    assert "<t>" in f, f


@pytest.mark.parametrize("s", ["tensão", "corrente", "texto bonito tt", "pt", "ta"])
def test_t_dentro_de_palavra_fica_com_a_ia(s):
    f, _ = P(s)
    assert "<t>" not in f, f


@pytest.mark.parametrize("s", ["I = 15∠−20° A", "5 + 2I", "I₃ = V/15", "Desenhe I a 20°", "de I →"])
def test_corrente_I_vira_simbolo(s):
    f, _ = P(s)
    assert "<I>" in f, f


@pytest.mark.parametrize("s", ["Imag", "Inversão", "KCL", "II"])
def test_I_de_palavra_fica_texto(s):
    f, _ = P(s)
    assert "<I>" not in f, f


# ---------------------------------------------------------------- j complexo
@pytest.mark.parametrize("s", ["j5", "−j0,1866", "0,05 − j0,0866", "1/(−j10)", "jω"])
def test_j_complexo_vira_simbolo(s):
    items = parse_inline(s, set())
    def has_j(its):
        for x in its:
            if isinstance(x, Sym) and x.ch == "j":
                return True
            if isinstance(x, Frac) and (has_j(x.num) or has_j(x.den)):
                return True
        return False
    assert has_j(items), flat(items)


@pytest.mark.parametrize("s", ["janela", "hoje", "jogo 2"])
def test_j_de_palavra_fica_texto(s):
    f, _ = P(s)
    assert "<j>" not in f


# ---------------------------------------------------------------- menos x hífen
@pytest.mark.parametrize("s,minus", [
    ("15∠−20°", True), ("−5 − j10", True), ("x - 1", True), ("=-5", True), ("(-16 + j40)", True),
    ("e-mail", False), ("bem-vindo", False), ("pré-requisito", False), ("Usa-se", False),
])
def test_menos_versus_hifen(s, minus):
    f, _ = P(s)
    assert ("<−>" in f) == minus, f


# ---------------------------------------------------------------- frações e afins
@pytest.mark.parametrize("s,frac", [
    ("5/4", True), ("x+5/2", True), ("dy/dx", True), ("311/√2", True), ("ω/2π", True),
    ("1/(5 + j8,66)", True), ("V₁/(−j2)", True), ("\\frac{a}{b}", True),
    ("rad/s", False), ("e/ou", False), ("km/h", False),
])
def test_fracoes(s, frac):
    items = parse_inline(s, set())
    assert any(isinstance(x, Frac) for x in items) == frac, flat(items)


def test_x_mais_cinco_meios_so_o_cinco_em_cima():
    f, _ = P("x+5/2")
    assert f[0] == "'x'" and "<+>" in f and f[-1] == "Frac('5' / '2')", f


def test_raiz_e_potencia():
    items = parse_inline("√(x+1) + x² + e^{2t}", set())
    assert any(isinstance(x, Root) for x in items)
    assert sum(isinstance(x, Sup) for x in items) == 2


def test_indices_unicode():
    items = parse_inline("V₁ + I₃ + Vₛ + Vₓ + Vₒ", set())
    assert sum(isinstance(x, Sub) for x in items) == 5


def test_integral_somatorio_limite():
    items = parse_inline("∫_0^1 x dx + ∑_{n=1}^{∞} 1/n² + lim_{x→0} sen x / x", set())
    assert sum(isinstance(x, BigOp) for x in items) == 3


def test_ordinal_e_check():
    f, bad = P("2º quadrante ✓")
    assert "Sup('o')" in f and "<✓>" in f and not bad


def test_latex_quebra_de_linha_na_matriz():
    items = parse_inline(r"\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}", set())
    m = [x for x in items if isinstance(x, Matrix)]
    assert m and len(m[0].rows) == 2 and len(m[0].rows[0]) == 2


# ---------------------------------------------------------------- matrizes x citações
@pytest.mark.parametrize("s,want", [
    ("(Agbo et al., 2025; Haindl & Weinberger, 2024)", False),
    ("(Silva, 2020; Souza, 2021)", False),
    ("{ por exemplo; outra coisa }", False),
    ("A = [1 2; 3 4]", True), ("B = (a, b; c, d)", True), ("|1 2; 3 4|", True),
    ("{ 2x + y = 5; x - y = 1 }", True), ("v = [1; 2; 3]", True),
])
def test_matriz_versus_citacao(s, want):
    items = parse_inline(s, set())
    assert any(isinstance(x, Matrix) for x in items) == want


# ---------------------------------------------------------------- nada se perde
def test_conversa_inteira_sem_caracteres_perdidos(conversa):
    rows, bad = plan_document(conversa, 40)
    assert bad == set()
    assert len(rows) > 100


def test_acentos_preservados():
    items = parse_inline("Admitância, impedância, ação, você, à, ç", set())
    txt = all_text(items)
    for w in ("Admitância", "impedância", "ação", "você", "à", "ç"):
        assert w in txt, txt


def test_caractere_desconhecido_e_reportado():
    _, bad = P("emoji 😀 aqui")
    assert "😀" in bad


@pytest.mark.parametrize("s", ["", " ", "((((", "))))", "/", "1/", "/2", "√", "^", "_", "[;]", "{}", "\\frac{",
                               "∫", "a" * 500, "j" * 50, "t" * 50, "−" * 30, "°°°", "|||", "1/2/3/4/5"])
def test_entradas_estranhas_nao_quebram(s):
    rows, _ = plan_document(s, 40)
    for r in rows:
        assert r.slots >= 1
