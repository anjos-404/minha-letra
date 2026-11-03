# Minha Letra

Escreve qualquer texto na **letra de uma pessoa**. Ela ensina a letra escrevendo
uma frase num quadro (dedo, caneta ou mouse) ou mandando uma foto do caderno, e
a IA escreve o texto colado no mesmo estilo, numa folha de caderno virtual.

## Destaques

- **Rede neural rodando em NumPy puro** (com PyTorch opcional para usar GPU), sem TensorFlow
- **Aprende o estilo por foto**: separa a tinta do papel, reconstrói o caminho da caneta e escolhe a reconstrução mais natural
- **Folha completa**: acentos do português, frações, expoentes, integrais, letras gregas, tabelas e diagramas de circuito
- **Naturalidade**: pressão da caneta, linha que ondula, inclinação e tamanho variando
- **Exporta** PNG, SVG e PDF (uma folha A4 por página)

## Tecnologias

Python · FastAPI · NumPy · SciPy · Pillow · PyTorch (opcional) · HTML/JavaScript · pytest · Playwright

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app
```

Abra http://localhost:8000.

Com placa NVIDIA, instale também `requirements-gpu.txt` para gerar bem mais rápido.

## Estrutura

```
app.py           servidor FastAPI (/api/photo, /api/generate, /api/line)
hand/model.py    a rede neural (LSTM + atenção) em NumPy
hand/photo.py    foto do caderno → traços de caneta
hand/layout.py   montagem da folha: texto, fórmulas, tabelas
hand/natural.py  variações humanas da escrita
web/index.html   interface
tests/           233 testes rápidos + testes de IA, servidor e navegador
```

## Testes

```bash
pip install pytest httpx
python -m pytest -m "not slow"   # rápidos (~5 s)
python -m pytest                 # tudo, incluindo IA e navegador (~6 min em CPU)
```

## Créditos e licença dos dados

Modelo de Alex Graves (2013, *Generating Sequences With Recurrent Neural Networks*).
Os pesos vêm do projeto [sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)
(MIT), treinado na base IAM On-Line, liberada para pesquisa: uso para estudo e
uso pessoal. `tools/export_weights.py` converteu o checkpoint original para `model/weights.npz`.

## Uso responsável

Não use para assinaturas, documentos oficiais ou para se passar por outra pessoa.
