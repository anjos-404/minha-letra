"""Servidor do protótipo: escreve qualquer texto na letra da pessoa.

Rodar:  python3 -m venv .venv && source .venv/bin/activate
        pip install -r requirements.txt
        uvicorn app:app --reload
Abrir:  http://localhost:8000
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from hand.model_torch import load_model          # GPU se tiver (PyTorch)
except ImportError:                                  # sem PyTorch: modo NumPy
    from hand.model import HandwritingModel

    def load_model(path):
        m = HandwritingModel(path)
        m.device_name = "CPU (NumPy) — instale o PyTorch para usar a GPU"
        return m
from hand.photo import photo_to_strokes
from hand.sample import distill_style, pick_best_variant
from hand.layout import (generate_rows, metrics_from_sample, paginate, plan_document, render_pages, row_body,
                         weak_rows)
from hand.strokes import prepare_user_sample, to_model_text

ROOT = os.path.dirname(os.path.abspath(__file__))
STYLES_DIR = os.path.join(ROOT, "model", "styles")
MAX_TEXT = 30000   # ~ 60 folhas de caderno
LINE_CHARS = 40
TRIES = 4          # versões geradas por linha; fica a melhor
MAX_PHOTO = 12_000_000  # ~12 MB em base64

app = FastAPI(title="Minha Letra")
model = load_model(os.path.join(ROOT, "model", "weights.npz"))
print(f"[minha-letra] rodando em: {model.device_name}", flush=True)


class Sample(BaseModel):
    text: str = Field(..., max_length=80)
    # cada traço: [[x, y, t_ms], ...] do quadro, ou [[x, y], ...] vindo de foto
    strokes: list[list[list[float]]] = []
    # amostra já pronta (vinda da leitura da foto): [[dx, dy, levantou], ...]
    offsets: list[list[float]] | None = Field(None, max_length=2400)


class GenerateRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT)
    sample: Sample | None = None
    style: int | None = Field(None, ge=0, le=12)
    neatness: float = Field(1.2, ge=0.0, le=3.0)  # "bias" do modelo: maior = mais caprichado
    naturalness: float = Field(1.0, ge=0.0, le=2.0)  # variação de linha, inclinação e pressão
    paper: str = "pautado"
    ink: str = "#1d2f7a"
    seed: int | None = None


class LineRequest(GenerateRequest):
    line_index: int = Field(..., ge=0, le=MAX_TEXT)   # texto longo tem centenas de linhas


class PhotoRequest(BaseModel):
    image: str = Field(..., max_length=MAX_PHOTO)  # data URL (PNG/JPEG)
    phrase: str = Field(..., max_length=80)        # o que está escrito na foto
    focus: list[float] | None = None               # [topo, base] da linha marcada (fração da altura)


def _builtin_style(i: int):
    s = np.load(os.path.join(STYLES_DIR, f"style-{i}-strokes.npy"), allow_pickle=False)
    c = np.load(os.path.join(STYLES_DIR, f"style-{i}-chars.npy"), allow_pickle=False).tobytes().decode()
    return s, c


def _prime(req: GenerateRequest):
    if req.sample and req.sample.offsets:
        off = np.asarray(req.sample.offsets, dtype=np.float32)
        if off.ndim != 2 or off.shape[1] != 3 or len(off) < 20 or not np.isfinite(off).all():
            raise HTTPException(400, "A amostra da foto está inválida. Leia a foto de novo.")
        off[:, 2] = (off[:, 2] > 0.5).astype(np.float32)
        off[:, :2] = np.clip(off[:, :2], -60, 60)
        text, _ = to_model_text(req.sample.text.strip())
        return off, text
    if req.sample and req.sample.strokes:
        prime_text, _ = to_model_text(req.sample.text.strip())
        if len(prime_text) < 8:
            raise HTTPException(400, "A frase de amostra precisa ter pelo menos 8 letras.")
        try:
            return prepare_user_sample(req.sample.strokes, n_chars=len(prime_text)), prime_text
        except ValueError:
            raise HTTPException(400, "A amostra está vazia. Escreva a frase no quadro ou envie uma foto.")
    return _builtin_style(req.style if req.style is not None else 9)


def _style_args(req: GenerateRequest):
    paper = req.paper if req.paper in ("pautado", "branco", "quadriculado") else "pautado"
    # só cor hexadecimal: a cor vai direto num atributo do SVG (nada de aspas ou tags)
    ink = req.ink if re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", req.ink or "") else "#1d2f7a"
    return paper, ink


PHOTO_TIPS = ("Dicas: escreva um pouco maior e com espaço entre as palavras, use caneta escura, "
              "tire a foto de cima com boa luz e marque a linha inteira, com as perninhas do g, p e q.")


@app.post("/api/photo")
def read_photo(req: PhotoRequest):
    """Foto de uma linha escrita -> amostra pronta para o modelo.

    1) reconstrói os traços da foto (várias versões possíveis);
    2) o modelo escolhe a versão que parece mais escrita de verdade;
    3) o modelo reescreve a frase nesse estilo várias vezes e fica com a
       reescrita legível mais parecida com a foto. Essa vira a amostra.
    """
    phrase, _ = to_model_text(req.phrase.strip())
    if len(phrase) < 8:
        raise HTTPException(400, "Escreva no campo a frase que está na foto (pelo menos 8 letras).")
    focus = None
    if req.focus and len(req.focus) == 2 and 0 <= req.focus[0] < req.focus[1] <= 1:
        focus = (float(req.focus[0]), float(req.focus[1]))
    try:
        res = photo_to_strokes(req.image, focus=focus)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(400, "Não consegui abrir essa imagem. Envie um JPG ou PNG.")

    t0 = time.time()
    best, _, ranked = pick_best_variant(model, res["variants"], phrase)
    off, text, info = distill_style(model, ranked[:3], phrase, candidates=10, seed=0)
    out = {"strokes": best, "width": res["width"], "height": res["height"],
           "seconds": round(time.time() - t0, 1), "ok": bool(info.get("distilled"))}
    if info.get("distilled"):
        out["sample"] = {"text": text, "offsets": np.round(off, 3).tolist()}
        if res["clipped"]:
            out["warning"] = "Parte da escrita encostou na borda da seleção. Se ficar estranho, marque com mais folga."
    else:
        out["warning"] = ("Consegui ler a foto, mas não aprendi sua letra bem o bastante: o texto pode sair confuso. "
                          + PHOTO_TIPS + " Ou escreva no quadro.")
    return out


def _unsupported_msg(bad: set) -> str | None:
    if not bad:
        return None
    chars = " ".join(sorted(bad))
    return f"Esses caracteres não sei escrever à mão e ficaram de fora: {chars}"


@app.post("/api/generate")
def generate(req: GenerateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "Escreva algum texto para transcrever.")
    prime, prime_text = _prime(req)
    metrics = metrics_from_sample(prime, prime_text)
    rows, bad = plan_document(text, LINE_CHARS, xh=metrics.xh)
    if not rows:
        raise HTTPException(400, "Não sobrou nada que eu saiba escrever nesse texto.")
    t0 = time.time()
    generate_rows(model, rows, prime=prime, prime_text=prime_text, bias=req.neatness, seed=req.seed, tries=TRIES)
    paper, ink = _style_args(req)
    pages = render_pages(rows, metrics, paper=paper, ink=ink, seed=req.seed, naturalness=req.naturalness)
    return {"pages": pages, "svg": pages[0], "lines": [r.kind for r in rows], "seconds": round(time.time() - t0, 1),
            "weak_lines": weak_rows(rows), "unsupported": _unsupported_msg(bad)}


@app.post("/api/line")
def regenerate_line(req: LineRequest):
    """Reescreve uma linha só (quando uma saiu ruim). Vale para linhas de tabela também."""
    prime, prime_text = _prime(req)
    metrics = metrics_from_sample(prime, prime_text)
    rows, _ = plan_document(req.text.strip(), LINE_CHARS, xh=metrics.xh)
    if req.line_index >= len(rows):
        raise HTTPException(400, "Essa linha não existe mais. Gere a folha de novo.")
    row = rows[req.line_index]
    generate_rows(model, [row], prime=prime, prime_text=prime_text, bias=req.neatness, seed=req.seed, tries=TRIES)
    start = paginate(rows)[req.line_index][1]     # pauta dentro da folha onde a linha está
    rng = np.random.default_rng(req.seed)
    return {"html": row_body(row, start, metrics, rng, nat=req.naturalness)}


@app.get("/api/info")
def info():
    return {"device": model.device_name}


@app.get("/")
def index():
    return FileResponse(os.path.join(ROOT, "web", "index.html"))


app.mount("/web", StaticFiles(directory=os.path.join(ROOT, "web")), name="web")
