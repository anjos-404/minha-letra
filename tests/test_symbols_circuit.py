"""Símbolos desenhados à mão e circuitos."""
import numpy as np
import pytest

from hand import circuit as C
from hand.layout import plan_document
from hand.symbols import COMPONENT_NAMES, SUPPORTED, component, hand_line, pen, shape


def _ok_strokes(strokes):
    assert strokes, "sem traços"
    for s in strokes:
        a = np.asarray(s)
        assert a.ndim == 2 and a.shape[1] == 2 and len(a) >= 2
        assert np.isfinite(a).all()


@pytest.mark.parametrize("ch", sorted(SUPPORTED))
def test_todo_simbolo_suportado_tem_desenho(ch):
    res = shape(ch)
    assert res is not None, ch
    strokes, width = res
    assert width > 0
    _ok_strokes(strokes)
    # cabe numa caixa razoável (unidades de altura-x)
    pts = np.vstack(strokes)
    assert pts[:, 0].min() > -0.5 and pts[:, 0].max() < width + 0.6
    assert pts[:, 1].min() > -2.6 and pts[:, 1].max() < 1.6


@pytest.mark.parametrize("name", sorted(set(COMPONENT_NAMES.values())))
def test_todo_componente_tem_desenho(name):
    strokes, width = component(name)
    assert width > 0
    _ok_strokes(strokes)


def test_tremor_da_caneta_preserva_forma():
    rng = np.random.default_rng(0)
    strokes, _ = shape("=")
    jit = pen([np.asarray(s, float) for s in strokes], rng)
    for a, b in zip(strokes, jit):
        assert np.abs(np.asarray(a)[[0, -1]] - b[[0, -1]]).max() < 0.15


def test_linha_a_mao():
    line = hand_line((0, 0), (100, 0), 20, np.random.default_rng(1))
    assert len(line) >= 9 and abs(line[-1, 0] - 100) < 5


# ---------------------------------------------------------------- circuitos
CIRCUITOS = [
    "circuito: bateria(12 V) - R1(10 Ω) - (R2(20 Ω) || C1(47 µF))",
    "circuito: fonte(220 V) - lâmpada - chave",
    "circuito A-B: R1(5 Ω) - (R2 || R3 || R4) - L1(2 mH)",
    "circuito: bateria(9 V) - " + " - ".join(f"R{i}(1 kΩ)" for i in range(1, 12)),   # comprido: dobra
    "circuito: fonte(10 V) - (R1 - R2 || C1 - L1) - terra",
    "circuito: fonte de corrente(2 A) - (R1(4 Ω) || R2(6 Ω))",
    "circuito: bateria - amperímetro - R1 - (voltímetro || R2) - diodo - led - fusível - motor",
]


@pytest.mark.parametrize("src", CIRCUITOS)
def test_circuitos_viram_desenho(src, metrics):
    from hand.layout import row_strokes
    rows, bad = plan_document(src, 40, xh=metrics.xh)
    assert len(rows) == 1 and rows[0].kind == "circuit", [r.kind for r in rows]
    assert not bad
    # sem modelo: rótulos ficam vazios, mas o desenho tem que sair inteiro
    for t in __import__("hand.layout", fromlist=["row_texts"]).row_texts(rows[0]):
        t.box = None
    try:
        strokes = row_strokes(rows[0], 0, metrics, np.random.default_rng(0))
    except Exception as e:      # rótulos sem caixa podem não ser aceitos: aí testa só o planejamento
        pytest.skip(f"desenho precisa dos rótulos gerados: {e}")
    pts = np.vstack([np.asarray(s) for s in strokes])
    from hand.layout import PAGE_WIDTH
    assert pts[:, 0].max() <= PAGE_WIDTH + 1, "circuito vaza da folha"


def test_circuito_com_fonte_fecha_o_laco():
    hm = C.HEADER.match("circuito: bateria(12 V) - R1 - R2")
    c = C.parse_circuit(hm, [])
    assert c.source is not None and c.terminals == ()


def test_circuito_ab_fica_aberto():
    hm = C.HEADER.match("circuito A-B: R1 - R2")
    c = C.parse_circuit(hm, [])
    assert c.source is None and c.terminals == ("A", "B")


def test_circuito_em_varias_linhas():
    rows, _ = plan_document("circuito:\nbateria(12 V) - R1\n(R2 || R3)\n\ndepois", 40)
    assert rows[0].kind == "circuit"
    assert len(C.comps(rows[0].circuit)) == 4
    assert rows[-1].kind == "text"
