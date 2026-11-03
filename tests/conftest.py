"""Configuração comum dos testes.

Rodar tudo:            python -m pytest
Só os rápidos:         python -m pytest -m "not slow"
Sem navegador:         python -m pytest -m "not ui"
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "tests", "data")
STYLES = os.path.join(ROOT, "model", "styles")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: usa o modelo de IA (alguns segundos a minutos)")
    config.addinivalue_line("markers", "ui: abre a página num navegador de verdade (Playwright)")


@pytest.fixture(scope="session")
def conversa():
    with open(os.path.join(DATA, "conversa.txt"), encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="session")
def style():
    """Estilo de letra pronto (o mesmo que o app usa quando não há amostra)."""
    s = np.load(os.path.join(STYLES, "style-9-strokes.npy"), allow_pickle=False)
    c = np.load(os.path.join(STYLES, "style-9-chars.npy"), allow_pickle=False).tobytes().decode()
    return s, c


@pytest.fixture(scope="session")
def metrics(style):
    from hand.layout import metrics_from_sample
    return metrics_from_sample(*style)


@pytest.fixture(scope="session")
def model():
    from hand.model_torch import load_model
    return load_model(os.path.join(ROOT, "model", "weights.npz"), prefer="cpu")
