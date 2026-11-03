"""O mesmo modelo de hand/model.py, em PyTorch: roda na GPU (NVIDIA/CUDA ou
Apple/MPS) quando tiver, e na CPU quando não tiver.

Mesma interface do HandwritingModel em NumPy: generate, generate_best,
judge_samples, legibility_batch... O resto do app não precisa saber qual está
rodando. Escolha com load_model() (variável MINHA_LETRA_DEVICE: auto, cuda,
mps, cpu ou numpy).
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .model import (K_ATTN, LSTM, M_OUT, N_CHARS, HandwritingModel, _drop_tiny_lead, _trim_lead, encode)


class TState:
    __slots__ = ("h1", "c1", "h2", "c2", "h3", "c3", "kappa", "w", "phi")

    def __init__(self, *a):
        for k, v in zip(self.__slots__, a):
            setattr(self, k, v)

    def astuple(self):
        return tuple(getattr(self, k) for k in self.__slots__)

    def where(self, mask, other: "TState") -> "TState":
        m = mask[:, None]
        return TState(*[torch.where(m, a, b) for a, b in zip(self.astuple(), other.astuple())])


class TorchHandwritingModel(HandwritingModel):
    def __init__(self, weights_path: str | None = None, device: str = "cpu"):
        super().__init__(weights_path)
        self.device = torch.device(device)
        t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.device)
        self.tk1, self.tb1 = t(self.k1), t(self.b1)
        self.tk2, self.tb2 = t(self.k2), t(self.b2)
        self.tk3, self.tb3 = t(self.k3), t(self.b3)
        self.tka, self.tba = t(self.ka), t(self.ba)
        self.tkg, self.tbg = t(self.kg), t(self.bg)
        if self.device.type == "cuda":
            self.device_name = "GPU " + torch.cuda.get_device_name(self.device)
        elif self.device.type == "mps":
            self.device_name = "GPU Apple (MPS)"
        else:
            self.device_name = f"CPU (PyTorch, {torch.get_num_threads()} núcleos)"

    # ---------- células ----------
    @staticmethod
    def _tlstm(x, h, c, kernel, bias):
        z = torch.cat([x, h], 1) @ kernel + bias
        i, j, f, o = z.chunk(4, 1)
        c_new = torch.sigmoid(f + 1.0) * c + torch.sigmoid(i) * torch.tanh(j)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def tzero(self, B: int, U: int) -> TState:
        z = lambda n: torch.zeros((B, n), dtype=torch.float32, device=self.device)
        return TState(z(LSTM), z(LSTM), z(LSTM), z(LSTM), z(LSTM), z(LSTM), z(K_ATTN), z(N_CHARS), z(U))

    def tstep(self, x, s: TState, onehot, char_mask, u) -> TState:
        h1, c1 = self._tlstm(torch.cat([s.w, x], 1), s.h1, s.c1, self.tk1, self.tb1)
        a = torch.nn.functional.softplus(torch.cat([s.w, x, h1], 1) @ self.tka + self.tba)
        alpha, beta, kappa = a.chunk(3, 1)
        kappa = s.kappa + kappa / 25.0
        beta = torch.clamp(beta, min=0.01)
        phi = (alpha[:, :, None] * torch.exp(-torch.square(kappa[:, :, None] - u) / beta[:, :, None])).sum(1)
        w = torch.einsum("bu,buc->bc", phi * char_mask, onehot)
        h2, c2 = self._tlstm(torch.cat([x, h1, w], 1), s.h2, s.c2, self.tk2, self.tb2)
        h3, c3 = self._tlstm(torch.cat([x, h2, w], 1), s.h3, s.c3, self.tk3, self.tb3)
        return TState(h1, c1, h2, c2, h3, c3, kappa, w, phi)

    def tsample(self, s: TState, bias, gen: torch.Generator):
        p = s.h3 @ self.tkg + self.tbg
        M = M_OUT
        pis, sig, rho, mu, e = p[:, :M], p[:, M:3 * M], p[:, 3 * M:4 * M], p[:, 4 * M:6 * M], p[:, 6 * M]
        pis = pis * (1.0 + bias[:, None])
        sig = sig - bias[:, None]
        pis = torch.softmax(pis, 1)
        pis = torch.where(pis < 0.01, torch.zeros_like(pis), pis)
        pis = pis / pis.sum(1, keepdim=True)
        sig = torch.clamp(torch.exp(sig), min=1e-4)
        rho = torch.clamp(torch.tanh(rho), -1 + 1e-6, 1 - 1e-6)
        e = torch.clamp(torch.sigmoid(e), 1e-8, 1 - 1e-8)
        e = torch.where(e < 0.01, torch.zeros_like(e), e)
        B = p.shape[0]
        k = torch.multinomial(pis, 1, generator=gen)[:, 0]
        ar = torch.arange(B, device=self.device)
        m1, m2 = mu[ar, k], mu[ar, M + k]
        s1, s2, r = sig[ar, k], sig[ar, M + k], rho[ar, k]
        z = torch.randn((B, 2), generator=gen, device=self.device)
        u = torch.rand((B,), generator=gen, device=self.device)
        x = torch.stack([m1 + s1 * z[:, 0], m2 + s2 * (r * z[:, 0] + torch.sqrt(1 - r * r) * z[:, 1]),
                         (u < e).float()], 1)
        return x

    def _onehot(self, encs: list[np.ndarray]):
        B, U = len(encs), max(len(e) for e in encs)
        onehot = torch.zeros((B, U, N_CHARS), dtype=torch.float32, device=self.device)
        mask = torch.zeros((B, U), dtype=torch.float32, device=self.device)
        for b, e in enumerate(encs):
            idx = torch.as_tensor(e, device=self.device)
            onehot[b, torch.arange(len(e), device=self.device), idx] = 1.0
            mask[b, :len(e)] = 1.0
        u = torch.arange(U, dtype=torch.float32, device=self.device)[None, None, :]
        return onehot, mask, u

    # ---------- geração ----------
    @torch.inference_mode()
    def tprimed(self, prime_strokes: np.ndarray, prefix: str) -> TState:
        key = (hash(prime_strokes.astype(np.float32).tobytes()), prefix, "torch")
        cache = self.__dict__.setdefault("_prime_cache", {})
        if key in cache:
            return cache[key]
        onehot, mask, u = self._onehot([encode(prefix)])
        s = self.tzero(1, onehot.shape[1])
        X = torch.as_tensor(prime_strokes.astype(np.float32), device=self.device)
        for t in range(X.shape[0]):
            s = self.tstep(X[t:t + 1], s, onehot, mask, u)
        delta = (len(prefix) - 1) - int(torch.argmax(s.phi[0]))
        if abs(delta) >= 1:
            s.kappa = s.kappa + float(delta)
        if len(cache) > 8:
            cache.clear()
        cache[key] = s
        return s

    @torch.inference_mode()
    def generate(self, texts, bias=0.75, prime_strokes=None, prime_text=None, seed=None, max_steps_per_char=40):
        B = len(texts)
        bias_np = np.asarray(bias if isinstance(bias, (list, tuple, np.ndarray)) else [bias] * B, dtype=np.float32)
        tbias = torch.as_tensor(bias_np, device=self.device)
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed) if seed is not None else int(np.random.default_rng().integers(2 ** 31)))

        prefix = (prime_text + " ") if prime_strokes is not None else ""
        encs = [encode(prefix + t) for t in texts]
        onehot, mask, u = self._onehot(encs)
        U = onehot.shape[1]
        lens = torch.as_tensor([len(e) for e in encs], device=self.device)
        offset = len(prefix)

        if prime_strokes is not None:
            p = self.tprimed(prime_strokes, prefix)
            s = TState(*[a.repeat(B, 1) for a in p.astuple()[:8]],
                       torch.zeros((B, U), dtype=torch.float32, device=self.device))
            x = self.tsample(s, tbias, gen)
        else:
            s = self.tzero(B, U)
            x = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).repeat(B, 1)

        max_steps = max_steps_per_char * max(len(t) for t in texts) + 20
        out = torch.zeros((max_steps, B, 3), dtype=torch.float32, device=self.device)
        idx = torch.zeros((max_steps, B), dtype=torch.long, device=self.device)
        done = torch.zeros(B, dtype=torch.bool, device=self.device)
        length = torch.full((B,), max_steps, dtype=torch.long, device=self.device)
        t = 0
        for t in range(max_steps):
            s_new = self.tstep(x, s, onehot, mask, u)
            s = s_new.where(~done, s) if t else s_new
            x_new = self.tsample(s, tbias, gen)
            ci = torch.argmax(s.phi, 1)
            out[t] = x_new
            idx[t] = ci
            fin = ((ci >= lens - 1) & (x_new[:, 2] == 1.0)) | (ci >= lens)
            newly = fin & ~done
            length = torch.where(newly, torch.full_like(length, t + 1), length)
            done = done | fin
            x = x_new
            if t % 16 == 15 and bool(done.all()):   # sincroniza com a CPU só de vez em quando
                break
        out_np = out[:t + 1].transpose(0, 1).cpu().numpy()
        idx_np = idx[:t + 1].transpose(0, 1).cpu().numpy()
        len_np = length.cpu().numpy()
        results = []
        for b in range(B):
            n = int(min(len_np[b], t + 1))
            results.append((out_np[b, :n].astype(np.float32), idx_np[b, :n] - offset))
        if prime_strokes is not None:
            results = [_trim_lead(o, i) for o, i in results]
            results = [_drop_tiny_lead(o, i) if tx[:1].isalnum() else (o, i) for (o, i), tx in zip(results, texts)]
        return results

    # ---------- o modelo julga amostras ----------
    def _tlog_density(self, s: TState, nxt):
        p = s.h3 @ self.tkg + self.tbg
        M = M_OUT
        pis, sig, rho, mu, e = p[:, :M], p[:, M:3 * M], p[:, 3 * M:4 * M], p[:, 4 * M:6 * M], p[:, 6 * M]
        logpi = torch.log_softmax(pis, 1)
        s1 = torch.clamp(torch.exp(sig[:, :M]), min=1e-4)
        s2 = torch.clamp(torch.exp(sig[:, M:]), min=1e-4)
        r = torch.clamp(torch.tanh(rho), -1 + 1e-6, 1 - 1e-6)
        dx = (nxt[:, 0:1] - mu[:, :M]) / s1
        dy = (nxt[:, 1:2] - mu[:, M:]) / s2
        z = dx * dx + dy * dy - 2 * r * dx * dy
        one_r = 1 - r * r
        logn = -torch.log(2 * np.pi * s1 * s2 * torch.sqrt(one_r)) - z / (2 * one_r)
        lmix = torch.logsumexp(logpi + logn, 1)
        pe = torch.clamp(torch.sigmoid(e), 1e-6, 1 - 1e-6)
        lber = torch.where(nxt[:, 2] == 1, torch.log(pe), torch.log(1 - pe))
        return lmix + lber

    @torch.inference_mode()
    def _run_judge(self, samples, encs, first_pen_up: bool):
        onehot, mask, u = self._onehot(encs)
        B, U = len(samples), onehot.shape[1]
        lens = torch.as_tensor([len(x) for x in samples], device=self.device)
        T = int(max(len(x) for x in samples))
        X = np.zeros((B, T, 3), np.float32)
        for b, x in enumerate(samples):
            X[b, :len(x)] = x
            if first_pen_up:
                X[b, 0] = [0.0, 0.0, 1.0]
        X = torch.as_tensor(X, device=self.device)
        s = self.tzero(B, U)
        total = torch.zeros(B, dtype=torch.float64, device=self.device) if self.device.type != "mps" \
            else torch.zeros(B, dtype=torch.float32, device=self.device)
        for t in range(T - 1):
            s_new = self.tstep(X[:, t], s, onehot, mask, u)
            alive = t < lens - 1
            s = s_new.where(alive, s) if t else s_new
            total += torch.where(alive, self._tlog_density(s, X[:, t + 1]), torch.zeros(B, device=self.device)).to(total.dtype)
        pos = torch.argmax(s.phi, 1).cpu().numpy()
        nll = (-total / torch.clamp(lens - 1, min=1)).cpu().numpy()
        return nll, pos

    def judge_samples(self, samples, text):
        nll, pos = self._run_judge(samples, [encode(text + " x")] * len(samples), first_pen_up=False)
        return [{"nll": float(nll[b]), "drift": int(abs(pos[b] - len(text)))} for b in range(len(samples))]

    def _judge_multi(self, samples, texts):
        nll, _ = self._run_judge(samples, [encode(t + " x") for t in texts], first_pen_up=True)
        return [float(v) for v in nll]


def load_model(weights_path: str | None = None, prefer: str | None = None):
    """Escolhe onde rodar: GPU NVIDIA > GPU Apple > CPU (PyTorch) > NumPy.

    MINHA_LETRA_DEVICE=cuda|mps|cpu|numpy força uma opção.
    """
    prefer = (prefer or os.environ.get("MINHA_LETRA_DEVICE", "auto")).lower()
    if prefer == "numpy":
        m = HandwritingModel(weights_path)
        m.device_name = "CPU (NumPy)"
        return m
    if prefer in ("auto", "cuda") and torch.cuda.is_available():
        return TorchHandwritingModel(weights_path, "cuda")
    if prefer in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return TorchHandwritingModel(weights_path, "mps")
    if prefer in ("cuda", "mps"):
        print(f"[minha-letra] {prefer} pedido, mas não está disponível: usando CPU")
    return TorchHandwritingModel(weights_path, "cpu")
