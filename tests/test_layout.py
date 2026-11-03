"""Quebra de linha, paginação e a folha em SVG (sem a IA: letras de mentira)."""
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from hand.layout import (LINE_HEIGHT, MARGIN_TOP, PAGE_SLOTS, PAGE_WIDTH, Box, Row, Space, Text, paginate,
                         parse_inline, plan_document, render_pages, row_strokes, row_texts, units, wrap)

SVG = "{http://www.w3.org/2000/svg}"
A4 = 297 / 210


def fake_generate(rows, metrics):
    """Põe uma 'letra' retangular em cada trecho, do tamanho que a IA faria."""
    for r in rows:
        for t in row_texts(r):
            w = max(len(t.text), 1) * metrics.char_w
            zig = np.array([[0, 0], [w * 0.5, -metrics.xh], [w, 0]], float)
            t.box = Box([zig], w, -metrics.xh, 0.0)
            t.legibility = 1.0
    return rows


def planned(text, metrics):
    rows, bad = plan_document(text, 40, xh=metrics.xh)
    return fake_generate(rows, metrics), bad


# ---------------------------------------------------------------- quebra de linha
def test_quebra_respeita_largura():
    items = parse_inline("palavra " * 60, set())
    for r in wrap(items, 40):
        assert units(r) <= 40 + 1e-6


def test_palavra_gigante_e_partida():
    rows = wrap(parse_inline("a" * 130, set()), 40)
    assert len(rows) >= 4
    for r in rows:
        assert units(r) <= 40 + 1e-6


def test_conta_nao_e_cortada_no_sinal():
    rows = wrap(parse_inline("texto " * 5 + "C = 47 μF " + "texto " * 5, set()), 40)
    for r in rows:
        last = r[-1]
        assert not (hasattr(last, "ch") and last.ch in "=+"), "linha terminando em = ou +"


# ---------------------------------------------------------------- paginação
def check_pages(rows, pos, page_slots=PAGE_SLOTS):
    used = {}
    for r, (p, st) in zip(rows, pos):
        if r.kind == "blank":
            continue
        assert st + r.slots <= max(page_slots, r.slots), f"linha passa do fim da folha {p}"
        for s in range(st, st + r.slots):
            assert (p, s) not in used, f"duas linhas na mesma pauta {(p, s)}"
            used[(p, s)] = r
    pages = [p for p, _ in pos]
    assert pages == sorted(pages), "ordem das folhas trocada"


def test_paginacao_da_conversa(conversa, metrics):
    rows, _ = planned(conversa, metrics)
    pos = paginate(rows)
    check_pages(rows, pos)
    assert max(p for p, _ in pos) + 1 >= 5


def test_tabela_nao_e_cortada(metrics):
    linhas = "\n".join(f"linha {i}" for i in range(15))
    tabela = "| a | b |\n|---|---|\n" + "\n".join(f"| {i} | {i * i} |" for i in range(8))
    rows, _ = planned(linhas + "\n" + tabela, metrics)
    pos = paginate(rows)
    check_pages(rows, pos)
    tpages = {p for r, (p, _) in zip(rows, pos) if r.kind == "table"}
    assert len(tpages) == 1, "tabela dividida em duas folhas"


def test_linha_em_branco_nao_abre_folha(metrics):
    rows, _ = planned("\n".join(["x"] * PAGE_SLOTS + [""] * 3 + ["fim"]), metrics)
    pos = paginate(rows)
    first_on_p1 = [r for r, (p, _) in zip(rows, pos) if p == 1]
    assert first_on_p1[0].kind != "blank" or pos[rows.index(first_on_p1[0])][1] == 0


# ---------------------------------------------------------------- SVG
def parse_svg(svg):
    root = ET.fromstring(svg)
    assert root.tag == SVG + "svg"
    return root


def test_folhas_sao_svg_validos_com_proporcao_a4(conversa, metrics):
    rows, _ = planned(conversa, metrics)
    pages = render_pages(rows, metrics, seed=1)
    assert len(pages) >= 5
    for svg in pages:
        root = parse_svg(svg)
        _, _, w, h = map(float, root.get("viewBox").split())
        assert w == PAGE_WIDTH
        assert abs(h / w - A4) < 0.08, f"folha fora da proporção A4: {h / w:.3f}"
        classes = [g.get("class") for g in root.iter(SVG + "g")]
        assert "fundo" in classes and "tinta" in classes


def test_cada_linha_aparece_uma_vez(conversa, metrics):
    rows, _ = planned(conversa, metrics)
    pages = render_pages(rows, metrics, seed=1)
    seen = []
    for svg in pages:
        seen += [int(x) for x in re.findall(r'data-line="(\d+)"', svg)]
    want = [i for i, r in enumerate(rows) if r.kind != "blank"]
    assert sorted(seen) == want


def test_tracos_dentro_da_folha(conversa, metrics):
    rows, _ = planned(conversa, metrics)
    pos = paginate(rows)
    rng = np.random.default_rng(0)
    for r, (_, st) in zip(rows, pos):
        for s in row_strokes(r, st, metrics, rng, nat=2.0):     # naturalidade no máximo
            assert np.isfinite(s).all()
            assert s[:, 0].min() >= 0 and s[:, 0].max() <= PAGE_WIDTH
            top = MARGIN_TOP + LINE_HEIGHT * st - LINE_HEIGHT
            assert s[:, 1].min() >= top - 1, "traço invade a linha de cima"


@pytest.mark.parametrize("paper", ["pautado", "quadriculado", "branco"])
def test_papeis(paper, metrics):
    rows, _ = planned("teste de papel", metrics)
    svg = render_pages(rows, metrics, paper=paper)[0]
    parse_svg(svg)
    n_lines = svg.count("<line")
    assert (n_lines == 0) == (paper == "branco")


def test_mesma_semente_mesma_folha(metrics):
    rows, _ = planned("Fasor V = 311∠135° V\nf = ω/2π = 60 Hz", metrics)
    a = render_pages(rows, metrics, seed=7)
    b = render_pages(rows, metrics, seed=7)
    assert a == b
