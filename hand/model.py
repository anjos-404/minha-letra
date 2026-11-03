"""Inferência do modelo de síntese de escrita à mão (Graves, 2013) em NumPy puro.

Reimplementa a rede do projeto sjvasquez/handwriting-synthesis (MIT):
3 camadas LSTM de 400 unidades, janela de atenção com 10 gaussianas sobre os
caracteres do texto e saída em mistura de 20 gaussianas bivariadas + fim de traço.

Cada passo gera um deslocamento (dx, dy) da caneta e um bit "levantou a caneta".
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

ALPHABET = [
    '\x00', ' ', '!', '"', '#', "'", '(', ')', ',', '-', '.',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
    '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
]
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
N_CHARS = len(ALPHABET)  # 73
LSTM = 400
K_ATTN = 10
M_OUT = 20

_P = "rnn__LSTMAttentionCell__"


def encode(text: str) -> np.ndarray:
    """Texto -> índices (caracteres fora do alfabeto viram 0) + terminador 0."""
    return np.array([CHAR_TO_IDX.get(c, 0) for c in text] + [0], dtype=np.int64)


def _wrong_text(t: str, rng: np.random.Generator) -> str:
    """Texto "errado" do mesmo tamanho, para medir legibilidade.

    Trechos longos: as mesmas letras embaralhadas. Curtos (números, siglas):
    cada letra/dígito trocado por outro do mesmo tipo (28 -> 61, x -> k).
    """
    if len(t) >= 6 and len(set(t)) > 1:
        return "".join(rng.permutation(list(t)))
    digits, lowers, uppers = "0123456789", "abcdefhiklmnorstuvwz", "ABDEFHIKLMNOPRSTUW"
    out = []
    for c in t:
        pool = digits if c.isdigit() else lowers if c.islower() else uppers if c.isupper() else None
        out.append(rng.choice([p for p in pool if p != c]) if pool else c)
    return "".join(out)


LEAD_LAG = 8  # a atenção anda ~8 pontos à frente da caneta


def _drop_tiny_lead(off: np.ndarray, idx: np.ndarray, max_diag: float = 3.5, max_strokes: int = 2):
    """Descarta traços minúsculos no começo (resto de vírgula/ponto da amostra)
    quando a linha começa com letra ou número."""
    for _ in range(max_strokes):
        ends = np.where(off[:, 2] == 1)[0]
        if len(ends) < 2:
            break
        e = ends[0]
        xy = np.cumsum(off[: e + 1, :2], axis=0)
        diag = float(np.linalg.norm(xy.max(0) - xy.min(0)))
        if diag >= max_diag:
            break
        off, idx = off[e + 1:].copy(), idx[e + 1:]
        off[0, :2] = 0.0
    return off, idx


def _trim_lead(off: np.ndarray, idx: np.ndarray):
    """Corta o rabisco que às vezes sai antes da primeira letra.

    Logo após ler a amostra, a "mão" do modelo ainda pode desenhar um resto
    dela. São os pontos antes de a atenção chegar à 1ª letra (+ o atraso da
    caneta). Traços que terminam inteiros nessa região são descartados; o
    traço que atravessa a fronteira perde só a parte anterior à 1ª letra.
    """
    if len(off) == 0 or not np.any(idx >= 0):
        return off, idx
    j = int(np.argmax(idx >= 0))
    if j == 0:
        return off, idx
    ends = np.where(off[:, 2] == 1)[0]
    cut, start = 0, 0
    for e in ends:
        n = e + 1 - start
        if e < j + LEAD_LAG or (n <= 6 and e < j + 3 * LEAD_LAG):
            cut = e + 1          # traço inteiro na sobra, ou pingo solto logo no começo
            start = e + 1
        else:
            break
    cut = max(cut, j) if cut < j else cut
    if cut >= len(off) - 2:
        return off, idx
    out = off[cut:].copy()
    out[0, :2] = 0.0             # começa do zero, sem o salto vindo da sobra
    return out, idx[cut:]


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _softplus(x):
    return np.logaddexp(0.0, x)


@dataclass
class State:
    h1: np.ndarray
    c1: np.ndarray
    h2: np.ndarray
    c2: np.ndarray
    h3: np.ndarray
    c3: np.ndarray
    kappa: np.ndarray
    w: np.ndarray
    phi: np.ndarray

    def where(self, mask: np.ndarray, other: "State") -> "State":
        """Mantém `self` onde mask=True, senão usa `other` (por amostra do lote)."""
        m = mask[:, None]
        return State(*[np.where(m, a, b) for a, b in zip(self.astuple(), other.astuple())])

    def astuple(self):
        return (self.h1, self.c1, self.h2, self.c2, self.h3, self.c3, self.kappa, self.w, self.phi)


class HandwritingModel:
    def __init__(self, weights_path: str | None = None):
        weights_path = weights_path or os.path.join(os.path.dirname(__file__), "..", "model", "weights.npz")
        W = np.load(weights_path, allow_pickle=False)
        f64 = lambda k: W[k].astype(np.float32)
        self.k1, self.b1 = f64(_P + "lstm_cell__kernel"), f64(_P + "lstm_cell__bias")
        self.k2, self.b2 = f64(_P + "lstm_cell_1__kernel"), f64(_P + "lstm_cell_1__bias")
        self.k3, self.b3 = f64(_P + "lstm_cell_2__kernel"), f64(_P + "lstm_cell_2__bias")
        self.ka, self.ba = f64(_P + "attention__weights"), f64(_P + "attention__biases")
        self.kg, self.bg = f64("rnn__gmm__weights"), f64("rnn__gmm__biases")

    # ---------- células ----------
    @staticmethod
    def _lstm(x, h, c, kernel, bias):
        z = np.concatenate([x, h], axis=1) @ kernel + bias
        i, j, f, o = np.split(z, 4, axis=1)
        c_new = _sigmoid(f + 1.0) * c + _sigmoid(i) * np.tanh(j)
        h_new = _sigmoid(o) * np.tanh(c_new)
        return h_new, c_new

    def zero_state(self, batch: int, char_len: int) -> State:
        z = lambda n: np.zeros((batch, n), dtype=np.float32)
        return State(z(LSTM), z(LSTM), z(LSTM), z(LSTM), z(LSTM), z(LSTM), z(K_ATTN), z(N_CHARS), z(char_len))

    def step(self, x, s: State, onehot, char_mask) -> State:
        """Um passo da rede. x: (B,3); onehot: (B,U,73); char_mask: (B,U)."""
        h1, c1 = self._lstm(np.concatenate([s.w, x], 1), s.h1, s.c1, self.k1, self.b1)

        a = _softplus(np.concatenate([s.w, x, h1], 1) @ self.ka + self.ba)
        alpha, beta, kappa = np.split(a, 3, axis=1)
        kappa = s.kappa + kappa / 25.0
        beta = np.maximum(beta, 0.01)
        u = np.arange(onehot.shape[1], dtype=np.float32)[None, None, :]
        phi = np.sum(alpha[:, :, None] * np.exp(-np.square(kappa[:, :, None] - u) / beta[:, :, None]), axis=1)
        w = np.einsum("bu,buc->bc", phi * char_mask, onehot)

        h2, c2 = self._lstm(np.concatenate([x, h1, w], 1), s.h2, s.c2, self.k2, self.b2)
        h3, c3 = self._lstm(np.concatenate([x, h2, w], 1), s.h3, s.c3, self.k3, self.b3)
        return State(h1, c1, h2, c2, h3, c3, kappa, w, phi)

    def sample_output(self, s: State, bias: np.ndarray, rng: np.random.Generator):
        """Amostra (dx, dy, fim_de_traço) da mistura de gaussianas."""
        p = s.h3 @ self.kg + self.bg
        M = M_OUT
        pis, sig, rho, mu, e = p[:, :M], p[:, M:3 * M], p[:, 3 * M:4 * M], p[:, 4 * M:6 * M], p[:, 6 * M:]
        pis = pis * (1.0 + bias[:, None])
        sig = sig - bias[:, None]
        pis = np.exp(pis - pis.max(1, keepdims=True))
        pis /= pis.sum(1, keepdims=True)
        pis = np.where(pis < 0.01, 0.0, pis)
        pis /= pis.sum(1, keepdims=True)
        sig = np.maximum(np.exp(sig), 1e-4)
        rho = np.clip(np.tanh(rho), -1 + 1e-8, 1 - 1e-8)
        e = np.clip(_sigmoid(e[:, 0]), 1e-8, 1 - 1e-8)
        e = np.where(e < 0.01, 0.0, e)

        B = p.shape[0]
        ar = np.arange(B)
        cum = np.cumsum(pis, axis=1)
        k = np.minimum((cum < rng.random((B, 1)) * cum[:, -1:]).sum(1), M - 1)
        m1, m2 = mu[ar, k], mu[ar, M + k]
        s1, s2, r = sig[ar, k], sig[ar, M + k], rho[ar, k]
        z1, z2 = rng.standard_normal(B), rng.standard_normal(B)
        out = np.empty((B, 3), dtype=np.float32)
        out[:, 0] = m1 + s1 * z1
        out[:, 1] = m2 + s2 * (r * z1 + np.sqrt(1 - r * r) * z2)
        out[:, 2] = (rng.random(B) < e).astype(np.float32)
        return out

    # ---------- geração ----------
    def primed_state(self, prime_strokes: np.ndarray, prefix: str) -> State:
        """Estado (lote de 1) depois de ler a amostra. Guardado em cache: gerar
        várias vezes com a mesma amostra (outra linha, reescrever) não relê."""
        key = (hash(prime_strokes.astype(np.float32).tobytes()), prefix)
        cache = self.__dict__.setdefault("_prime_cache", {})
        if key in cache:
            return cache[key]
        enc = encode(prefix)
        U = len(enc)
        onehot = np.zeros((1, U, N_CHARS), np.float32)
        onehot[0, np.arange(U), enc] = 1.0
        mask = np.ones((1, U), np.float32)
        s = self.zero_state(1, U)
        for x in prime_strokes.astype(np.float32):
            s = self.step(x[None], s, onehot, mask)
        # Reposiciona a atenção no espaço antes do texto novo. Com amostras
        # imperfeitas (foto, captura irregular) a atenção pode terminar
        # algumas letras antes do fim da amostra, e o modelo recopiaria o
        # final dela ou se perderia. Com uma amostra boa o ajuste é ~0.
        delta = (len(prefix) - 1) - int(np.argmax(s.phi[0]))
        if abs(delta) >= 1:
            s.kappa = s.kappa + np.float32(delta)
        if len(cache) > 8:
            cache.clear()
        cache[key] = s
        return s

    def generate(
        self,
        texts: list[str],
        bias: float | list[float] = 0.75,
        prime_strokes: np.ndarray | None = None,
        prime_text: str | None = None,
        seed: int | None = None,
        max_steps_per_char: int = 40,
    ):
        """Gera traços para cada linha.

        prime_strokes: deslocamentos (N,3) normalizados de uma amostra de escrita
        cuja transcrição é prime_text. O modelo "lê" essa amostra antes de escrever,
        e com isso continua no mesmo estilo (priming).

        Retorna lista de (offsets (T,3), char_idx (T,)), onde char_idx diz em qual
        caractere do texto-alvo a atenção estava em cada ponto (usado p/ acentos).
        """
        rng = np.random.default_rng(seed)
        B = len(texts)
        bias = np.asarray(bias if isinstance(bias, (list, tuple, np.ndarray)) else [bias] * B, dtype=np.float32)

        prefix = (prime_text + " ") if prime_strokes is not None else ""
        encoded = [encode(prefix + t) for t in texts]
        U = max(len(c) for c in encoded)
        onehot = np.zeros((B, U, N_CHARS), dtype=np.float32)
        char_mask = np.zeros((B, U), dtype=np.float32)
        lens = np.array([len(c) for c in encoded])
        for b, c in enumerate(encoded):
            onehot[b, np.arange(len(c)), c] = 1.0
            char_mask[b, :len(c)] = 1.0
        offset = len(prefix)

        s = self.zero_state(B, U)
        if prime_strokes is not None:
            # A amostra é igual para todas as linhas/tentativas: lê uma vez só
            # (lote de 1) e copia o estado para o lote inteiro.
            p = self.primed_state(prime_strokes, prefix)
            s = State(*[np.repeat(a, B, axis=0) for a in p.astuple()[:8]], np.zeros((B, U), np.float32))
            x = self.sample_output(s, bias, rng)
        else:
            x = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (B, 1))

        max_steps = max_steps_per_char * max(len(t) for t in texts) + 20
        outs = [[] for _ in range(B)]
        idxs = [[] for _ in range(B)]
        done = np.zeros(B, dtype=bool)
        for _ in range(max_steps):
            s_new = self.step(x, s, onehot, char_mask)
            s = s.where(done, s_new) if done.any() else s_new
            x_new = self.sample_output(s, bias, rng)
            char_idx = np.argmax(s.phi, axis=1)
            for b in range(B):
                if done[b]:
                    continue
                outs[b].append(x_new[b].copy())
                idxs[b].append(int(char_idx[b]) - offset)
                final_char = char_idx[b] >= lens[b] - 1
                past_final = char_idx[b] >= lens[b]
                if (final_char and x_new[b, 2] == 1.0) or past_final:
                    done[b] = True
            x = x_new
            if done.all():
                break

        results = [(np.array(o, dtype=np.float32).reshape(-1, 3), np.array(i)) for o, i in zip(outs, idxs)]
        if prime_strokes is not None:
            results = [_trim_lead(o, i) for o, i in results]
            results = [_drop_tiny_lead(o, i) if t[:1].isalnum() else (o, i)
                       for (o, i), t in zip(results, texts)]
        return results

    # ---------- o modelo julga amostras ----------
    def _log_density(self, s: State, nxt: np.ndarray) -> np.ndarray:
        """log p(próximo ponto | estado), sem 'capricho' (bias 0)."""
        p = s.h3 @ self.kg + self.bg
        M = M_OUT
        pis, sig, rho, mu, e = p[:, :M], p[:, M:3 * M], p[:, 3 * M:4 * M], p[:, 4 * M:6 * M], p[:, 6 * M]
        logpi = pis - pis.max(1, keepdims=True)
        logpi = logpi - np.log(np.exp(logpi).sum(1, keepdims=True))
        s1, s2 = np.maximum(np.exp(sig[:, :M]), 1e-4), np.maximum(np.exp(sig[:, M:]), 1e-4)
        r = np.clip(np.tanh(rho), -1 + 1e-6, 1 - 1e-6)
        dx = (nxt[:, 0:1] - mu[:, :M]) / s1
        dy = (nxt[:, 1:2] - mu[:, M:]) / s2
        z = dx * dx + dy * dy - 2 * r * dx * dy
        one_r = 1 - r * r
        logn = -np.log(2 * np.pi * s1 * s2 * np.sqrt(one_r)) - z / (2 * one_r)
        a = logpi + logn
        amax = a.max(1, keepdims=True)
        lmix = (amax + np.log(np.exp(a - amax).sum(1, keepdims=True)))[:, 0]
        pe = np.clip(_sigmoid(e), 1e-6, 1 - 1e-6)
        lber = np.where(nxt[:, 2] == 1, np.log(pe), np.log(1 - pe))
        return lmix + lber

    def judge_samples(self, samples: list[np.ndarray], text: str) -> list[dict]:
        """Lê várias versões da mesma amostra (num lote só) e diz quão
        "natural" cada uma parece para o modelo.

        nll: surpresa média por ponto (menor = parece mais escrita de verdade).
        drift: quantas letras a atenção ficou longe do fim da frase ao terminar.
        """
        B = len(samples)
        enc = encode(text + " x")
        U = len(enc)
        onehot = np.zeros((B, U, N_CHARS), np.float32)
        onehot[:, np.arange(U), enc] = 1.0
        mask = np.ones((B, U), np.float32)
        lens = np.array([len(x) for x in samples])
        T = int(lens.max())
        X = np.zeros((B, T, 3), np.float32)
        for b, x in enumerate(samples):
            X[b, :len(x)] = x
        s = self.zero_state(B, U)
        total = np.zeros(B)
        for t in range(T - 1):
            s_new = self.step(X[:, t], s, onehot, mask)
            alive = t < lens - 1
            s = s_new if alive.all() else s_new.where(alive, s)
            lp = self._log_density(s, X[:, t + 1])
            total += np.where(alive, lp, 0.0)
        pos = np.argmax(s.phi, axis=1)
        return [{"nll": float(-total[b] / max(lens[b] - 1, 1)), "drift": int(abs(pos[b] - len(text)))}
                for b in range(B)]

    # ---------- várias tentativas, fica a melhor ----------
    @staticmethod
    def line_quality(text: str, char_idx: np.ndarray, n_points: int, max_steps_per_char: int = 40) -> float:
        """Penalidade (menor = melhor) de uma linha gerada, sem OCR.

        Numa linha boa a atenção anda de letra em letra num ritmo regular.
        Quando o modelo se perde, ele empaca numa letra (rabisco longo),
        pula letras ou não termina.
        """
        L = len(text)
        if L == 0:
            return 0.0
        idx = char_idx[(char_idx >= 0) & (char_idx < L)]
        counts = np.bincount(idx, minlength=L)[:L].astype(float)
        letters = np.array([c != " " for c in text])
        med = max(np.median(counts[letters]) if letters.any() else 1.0, 1.0)
        pen = 0.0
        pen += 3.0 * np.sum((counts == 0) & letters)                       # letra pulada
        pen += np.sum(np.clip(counts / med - 3.0, 0, None))                # letra empacada
        per_char = n_points / L
        pen += 2.0 * max(0.0, per_char - 45) / 10 + 2.0 * max(0.0, 12 - per_char) / 4
        if n_points >= max_steps_per_char * L:                              # não terminou
            pen += 10.0
        back = np.sum(np.diff(char_idx) < -1)                               # atenção voltando
        pen += 0.5 * back
        return float(pen)

    def generate_best(self, texts: list[str], tries: int = 3, **kw):
        """Gera `tries` versões de cada linha num lote só e fica com a melhor."""
        if tries <= 1:
            return self.generate(texts, **kw)
        bias = kw.pop("bias", 0.75)
        batch = [t for t in texts for _ in range(tries)]
        bias_b = np.repeat(np.asarray(bias if isinstance(bias, (list, tuple, np.ndarray)) else [bias] * len(texts),
                                      dtype=np.float32), tries)
        outs = self.generate(batch, bias=bias_b, **kw)
        # legibilidade de cada tentativa: o texto certo explica os traços melhor
        # que o mesmo texto embaralhado? (ver hand/sample.py: legibility)
        rng = np.random.default_rng(kw.get("seed"))
        right_txt = batch
        wrong_txt = [_wrong_text(t, rng) for t in batch]
        margins = self.legibility_batch([o for o, _ in outs], right_txt, wrong_txt)
        best, self.last_legibility = [], []
        for i, t in enumerate(texts):
            cands = outs[i * tries:(i + 1) * tries]
            mg = margins[i * tries:(i + 1) * tries]
            q = [self.line_quality(t, ci, len(o)) for o, ci in cands]
            # linhas curtas demais: a margem é pouco confiável, vale o medidor antigo
            score = [(-m + 0.05 * qq) if len(t.strip()) >= 2 else (-0.5 * m + qq) for m, qq in zip(mg, q)]
            k = int(np.argmin(score))
            best.append(cands[k])
            self.last_legibility.append(float(mg[k]) if len(t.strip()) >= 6 else None)
        return best

    def legibility_batch(self, offsets: list[np.ndarray], right: list[str], wrong: list[str]) -> list[float]:
        """Margem de legibilidade de cada amostra com seu próprio texto (num lote)."""
        if not offsets:
            return []
        both = self._judge_multi(list(offsets) * 2, list(right) + list(wrong))  # um lote só
        n = len(offsets)
        return [b - a for a, b in zip(both[:n], both[n:])]

    def _judge_multi(self, samples: list[np.ndarray], texts: list[str]) -> list[float]:
        """Como judge_samples, mas cada amostra com o seu texto. Devolve a NLL média."""
        B = len(samples)
        encs = [encode(t + " x") for t in texts]
        U = max(len(e) for e in encs)
        onehot = np.zeros((B, U, N_CHARS), np.float32)
        mask = np.zeros((B, U), np.float32)
        for b, e in enumerate(encs):
            onehot[b, np.arange(len(e)), e] = 1.0
            mask[b, :len(e)] = 1.0
        lens = np.array([len(x) for x in samples])
        T = int(lens.max())
        X = np.zeros((B, T, 3), np.float32)
        for b, x in enumerate(samples):
            X[b, :len(x)] = x
            X[b, 0] = [0.0, 0.0, 1.0]
        s = self.zero_state(B, U)
        total = np.zeros(B)
        for t in range(T - 1):
            s_new = self.step(X[:, t], s, onehot, mask)
            alive = t < lens - 1
            s = s_new if alive.all() else s_new.where(alive, s)
            total += np.where(alive, self._log_density(s, X[:, t + 1]), 0.0)
        return [float(-total[b] / max(lens[b] - 1, 1)) for b in range(B)]
