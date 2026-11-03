"""O servidor (FastAPI): respostas, validação e segurança das entradas."""
import base64
import io
import os
import re
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow       # carrega o modelo e gera de verdade


@pytest.fixture(scope="module")
def client():
    import app as appmod
    return TestClient(appmod.app)


def gen(client, text, **kw):
    body = {"text": text, "style": 9, "seed": 3}
    body.update(kw)
    return client.post("/api/generate", json=body)


def test_info(client):
    r = client.get("/api/info")
    assert r.status_code == 200 and r.json()["device"]


def test_pagina_inicial(client):
    r = client.get("/")
    assert r.status_code == 200 and "Escrever na folha" in r.text


def test_gera_folha(client):
    r = gen(client, "Fasor: V = 311∠135° V\nf = ω/2π = 60 Hz")
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["pages"]) == 1 and d["svg"] == d["pages"][0]
    ET.fromstring(d["svg"])
    assert d["unsupported"] is None
    assert d["lines"] == ["text", "text"]
    assert 'class="tinta"' in d["svg"]


def test_texto_vazio(client):
    assert gen(client, "   ").status_code == 400


def test_texto_grande_demais(client):
    assert gen(client, "a" * 30001).status_code == 422


def test_caractere_desconhecido_e_avisado(client):
    d = gen(client, "oi 😀").json()
    assert d["unsupported"] and "😀" in d["unsupported"]


@pytest.mark.parametrize("ink", ['#"/>', '#"/><script>alert(1)</script>', "red", "#12", "#zzzzzz", "#1d2f7a"])
def test_cor_da_caneta_nao_injeta_codigo(client, ink):
    d = gen(client, "teste", ink=ink).json()
    ET.fromstring(d["svg"])                                   # continua XML válido
    assert "<script" not in d["svg"]
    used = re.search(r'class="tinta" fill="none" stroke="([^"]*)"', d["svg"]).group(1)
    assert re.fullmatch(r"#[0-9a-fA-F]{3}([0-9a-fA-F]{3})?", used), used


def test_papel_invalido_vira_pautado(client):
    d = gen(client, "teste", paper="<x>").json()
    assert "<x>" not in d["svg"] and d["svg"].count("<line") > 5


def test_reescrever_linha(client):
    text = "linha um\nlinha dois\nlinha três"
    r = client.post("/api/line", json={"text": text, "style": 9, "line_index": 1, "seed": 5})
    assert r.status_code == 200
    ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{r.json()["html"]}</svg>')


def test_reescrever_linha_inexistente(client):
    r = client.post("/api/line", json={"text": "só uma", "style": 9, "line_index": 5})
    assert r.status_code == 400


def test_reescrever_linha_de_texto_longo(client):
    """Texto grande (o limite é 30 mil letras) tem bem mais de 200 linhas: todas têm que poder ser reescritas."""
    text = "\n".join(f"{i}" for i in range(400))
    r = client.post("/api/line", json={"text": text, "style": 9, "line_index": 350, "seed": 1})
    assert r.status_code == 200, r.text


def test_reescrever_linha_na_segunda_folha_fica_na_pauta_certa(client):
    from hand.layout import PAGE_SLOTS, rule_y
    text = "\n".join(f"linha {i}" for i in range(PAGE_SLOTS + 3))
    i = PAGE_SLOTS + 1                                 # 2ª linha da 2ª folha
    html = client.post("/api/line", json={"text": text, "style": 9, "line_index": i, "seed": 1}).json()["html"]
    ys = [float(y) for y in re.findall(r"[ML]\s*[-\d.]+[ ,]([-\d.]+)", html)]
    assert ys and abs(sorted(ys)[len(ys) // 2] - rule_y(1)) < 40


def test_amostra_curta_demais(client):
    r = gen(client, "oi", sample={"text": "oi", "strokes": [[[0, 0, 0], [5, 5, 10]]]})
    assert r.status_code == 400


def test_amostra_de_foto_invalida(client):
    r = gen(client, "oi", sample={"text": "frase da foto", "offsets": [[1, 1, 0]] * 5})
    assert r.status_code == 400
    r = gen(client, "oi", sample={"text": "frase da foto", "offsets": [[1, 1]] * 40})
    assert r.status_code == 400


def test_amostra_do_quadro(client):
    """Simula alguém escrevendo no quadro (traços com tempo)."""
    import numpy as np
    strokes, x = [], 0
    for k in range(14):                                 # 14 "letras" em zigue-zague
        t = np.linspace(0, 1, 12)
        pts = [[x + 12 * tt, 30 - 18 * abs(np.sin(tt * 3.1)), k * 200 + 15 * j] for j, tt in enumerate(t)]
        strokes.append(pts)
        x += 16
    r = gen(client, "teste com amostra", sample={"text": "a frase do quadro", "strokes": strokes})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------- foto
def _photo(text="Fiz jogo de bola", size=(900, 160)):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for y in range(20, size[1], 36):                    # pauta do caderno
        d.line([(0, y), (size[0], y)], fill=(160, 190, 230), width=1)
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        if os.path.exists(f):
            font = ImageFont.truetype(f, 54)
    d.text((30, 40), text, fill=(20, 30, 90), font=font)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_foto_vira_amostra(client):
    r = client.post("/api/photo", json={"image": _photo(), "phrase": "Fiz jogo de bola"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["strokes"] and "ok" in d
    if d["ok"]:
        assert len(d["sample"]["offsets"]) >= 20
        assert gen(client, "teste", sample=d["sample"]).status_code == 200


def test_foto_frase_curta(client):
    assert client.post("/api/photo", json={"image": _photo(), "phrase": "oi"}).status_code == 400


@pytest.mark.parametrize("img", ["data:image/png;base64,AAAA", "não é imagem", "data:image/png;base64,"])
def test_foto_invalida(client, img):
    r = client.post("/api/photo", json={"image": img, "phrase": "Fiz jogo de bola"})
    assert r.status_code == 400


def test_foto_em_branco(client):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (600, 200), "white").save(buf, "PNG")
    img = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    r = client.post("/api/photo", json={"image": img, "phrase": "Fiz jogo de bola"})
    assert r.status_code == 400
