"""A página no navegador: escrever, várias folhas, reescrever linha, exportar e imprimir."""
import io
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytestmark = [pytest.mark.ui, pytest.mark.slow]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = dict(os.environ, MINHA_LETRA_DEVICE="cpu")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port", str(port)], cwd=ROOT, env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(url + "/api/info", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        p.kill()
        pytest.fail("servidor não subiu")
    yield url
    p.terminate()
    p.wait(10)


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, server):
    ctx = browser.new_context(accept_downloads=True, viewport={"width": 1300, "height": 900})
    pg = ctx.new_page()
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    # fontes do Google podem não carregar sem internet: isso não é erro do app
    pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text
          and errors.append(m.text))
    pg.goto(server)
    pg.errors = errors
    yield pg
    ctx.close()


def write(pg, text, timeout=240_000):
    pg.fill("#text", text)
    pg.click("#go")
    pg.wait_for_selector("#actions2.show", timeout=timeout)
    pg.evaluate("document.getAnimations().forEach(a => a.finish())")


def downloads(pg, button, n):
    got = []
    with pg.expect_download() as first:
        pg.click(button)
    got.append(first.value)
    for _ in range(n - 1):
        got.append(pg.wait_for_event("download", timeout=10_000))
    return got


LONGO = "\n".join(f"Linha {i}: V = 311∠135° V e v(t) = 311 cos(377t)" for i in range(24))


def test_pagina_abre_sem_erros(page):
    page.wait_for_timeout(500)
    assert page.title()
    assert not page.errors, page.errors


def test_escreve_e_mostra_botoes(page):
    write(page, "Fasor: V = 311∠135° V")
    assert page.locator("#sheet .page").count() == 1
    assert page.locator("#sheet g.linha").count() == 1
    assert page.is_visible("#dlCleanPng") and page.is_visible("#print")
    assert not page.errors, page.errors


def test_varias_folhas_e_reescrever_linha_da_segunda(page):
    write(page, LONGO)
    n = page.locator("#sheet .page").count()
    assert n >= 2
    g = page.locator("#sheet .page").nth(1).locator("g.linha").first
    idx = g.get_attribute("data-line")
    before = g.inner_html()
    g.click()
    page.wait_for_function(f"!document.querySelector('g.linha[data-line=\"{idx}\"]').classList.contains('busy')",
                           timeout=60_000)
    after = page.locator(f'g.linha[data-line="{idx}"]').inner_html()
    assert after != before, "linha não foi reescrita"
    assert page.locator(f'g.linha[data-line="{idx}"]').count() == 1
    assert "Não consegui" not in page.inner_text("#status")
    assert not page.errors, page.errors


def test_exporta_png_e_svg_de_cada_folha(page, tmp_path):
    write(page, LONGO)
    n = page.locator("#sheet .page").count()
    pngs = downloads(page, "#dlPng", n)
    names = sorted(d.suggested_filename for d in pngs)
    assert names == [f"minha-letra-{k:02d}.png" for k in range(1, n + 1)]


def test_exporta_sem_fundo(page, tmp_path):
    from PIL import Image
    write(page, "Y = 1/(5 + j8,66) = 0,1∠−60° S")
    (png,) = downloads(page, "#dlCleanPng", 1)
    f = tmp_path / "a.png"
    png.save_as(f)
    im = Image.open(f)
    assert im.mode == "RGBA"
    alpha = im.getchannel("A")
    transp = sum(1 for a in alpha.getdata() if a == 0) / (im.width * im.height)
    assert 0.6 < transp < 0.995, transp                   # fundo transparente, mas tem tinta
    w, h = im.size
    assert w < 900 * 4 * 0.98 and h < 600 * 4, "não recortou no que foi escrito"
    bbox = alpha.getbbox()
    assert bbox[0] < w * 0.1 and bbox[2] > w * 0.9       # a escrita encosta (quase) nas bordas: recorte justo

    (svg,) = downloads(page, "#dlCleanSvg", 1)
    f2 = tmp_path / "a.svg"
    svg.save_as(f2)
    txt = f2.read_text()
    assert 'class="fundo"' not in txt and 'class="hit"' not in txt and "<path" in txt
    assert "tabindex" not in txt and "style=" not in txt


def test_imprimir_uma_folha_por_pagina(page, tmp_path):
    write(page, LONGO)
    n = page.locator("#sheet .page").count()
    page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
    page.emulate_media(media="print")
    pdf = tmp_path / "f.pdf"
    page.pdf(path=str(pdf), prefer_css_page_size=True)
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    assert int(re.search(r"Pages:\s+(\d+)", out).group(1)) == n
    assert "(A4)" in out
    # nenhuma página em branco: todas têm tinta azul
    subprocess.run(["pdftoppm", "-r", "30", "-png", str(pdf), str(tmp_path / "p")], check=True)
    from PIL import Image
    for f in sorted(tmp_path.glob("p-*.png")):
        im = Image.open(f).convert("RGB")
        ink = sum(1 for r, g, b in im.getdata() if b > 80 and r < 90 and g < 110)
        assert ink > 50, f"{f.name} sem escrita"


def test_imprimir_no_meio_da_animacao_sai_completo(page, tmp_path):
    page.fill("#text", "Questão 1: Tempo para Fasor, usando a identidade do seno")
    page.click("#go")
    page.wait_for_selector("#actions2.show", timeout=120_000)
    # sem esperar a animação terminar
    page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
    hidden = page.evaluate("[...document.querySelectorAll('#sheet path')].filter(p => getComputedStyle(p).opacity === '0').length")
    assert hidden == 0


def test_celular_sem_rolagem_lateral(browser, server):
    ctx = browser.new_context(viewport={"width": 390, "height": 800})
    pg = ctx.new_page()
    pg.goto(server)
    assert pg.evaluate("document.documentElement.scrollWidth") <= 391
    ctx.close()


def test_quadro_aceita_escrita(page):
    pad = page.locator("#pad")
    box = pad.bounding_box()
    for k in range(10):
        x = box["x"] + 20 + k * 30
        page.mouse.move(x, box["y"] + 90)
        page.mouse.down()
        for j in range(8):
            page.mouse.move(x + j * 3, box["y"] + 90 - 40 * abs(((j % 4) - 2) / 2), steps=2)
        page.mouse.up()
    assert page.locator("#padEmpty").is_hidden()
    page.click("#undo")
    assert not page.errors, page.errors
