"""A IA escrevendo de verdade: legibilidade, tamanho das letras e consistência."""
import numpy as np
import pytest

from hand.layout import (PAGE_WIDTH, generate_rows, paginate, plan_document, render_pages, row_strokes, row_texts,
                         weak_rows)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def questao1(conversa, style, metrics, model):
    """As linhas da questão 1 e 2 da conversa (tem fasor, j, t, fração, √, ✓, 2º...)."""
    txt = conversa.split("3) Análise nodal")[0]
    rows, bad = plan_document(txt, 40, xh=metrics.xh)
    generate_rows(model, rows, prime=style[0], prime_text=style[1], bias=1.2, seed=11)
    return rows, bad


def test_nada_perdido(questao1):
    assert questao1[1] == set()


def test_todo_trecho_foi_escrito(questao1):
    rows, _ = questao1
    for r in rows:
        for t in row_texts(r):
            if t.text.strip():
                assert t.box is not None and t.box.strokes, t.text
                for s in t.box.strokes:
                    assert np.isfinite(s).all()


def test_largura_condiz_com_o_texto(questao1, metrics):
    """Trecho rabiscado ou cortado no meio fica largo/estreito demais para o número de letras."""
    rows, _ = questao1
    ratios = []
    for r in rows:
        for t in row_texts(r):
            n = len(t.base.strip())
            if n >= 4:
                ratios.append((t.box.w / (n * metrics.char_w), t.text))
    vals = np.array([v for v, _ in ratios])
    med = np.median(vals)
    off = [(round(v / med, 2), t) for v, t in ratios if not 0.45 < v / med < 2.1]
    assert len(off) <= max(1, len(ratios) // 30), off


def test_legibilidade(questao1):
    rows, _ = questao1
    leg = [t.legibility for r in rows for t in row_texts(r) if t.legibility is not None]
    assert np.median(leg) > 0.5
    assert len(weak_rows(rows)) <= max(1, len(rows) // 15), weak_rows(rows)


def test_folhas_finais(questao1, metrics):
    rows, _ = questao1
    pos = paginate(rows)
    rng = np.random.default_rng(0)
    for r, (_, st) in zip(rows, pos):
        for s in row_strokes(r, st, metrics, rng, nat=1.0):
            assert s[:, 0].max() <= PAGE_WIDTH and s[:, 0].min() >= 0
    pages = render_pages(rows, metrics, seed=11)
    assert 2 <= len(pages) <= 4


def test_mesma_semente_mesmo_resultado(style, model):
    a = model.generate_best(["Fasor 311"], tries=2, bias=1.2, prime_strokes=style[0], prime_text=style[1], seed=4)
    b = model.generate_best(["Fasor 311"], tries=2, bias=1.2, prime_strokes=style[0], prime_text=style[1], seed=4)
    assert np.allclose(a[0][0], b[0][0])


def test_pytorch_igual_ao_numpy(style):
    """O caminho rápido (PyTorch/GPU) calcula o mesmo que a versão de referência (NumPy)."""
    import os
    from hand.model import HandwritingModel
    from hand.model_torch import load_model
    w = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "weights.npz")
    ref, fast = HandwritingModel(w), load_model(w, prefer="cpu")
    off = style[0][:300].astype(np.float32)          # um pedaço de escrita real
    texts = [style[1][:12], "outra coisa"]
    a = np.asarray(ref._judge_multi([off, off], texts), float)
    b = np.asarray(fast._judge_multi([off, off], texts), float)
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3), (a, b)
    assert a[0] < a[1]                               # o texto certo é o mais provável
