
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

import torch


@dataclass
class FitHistoryEntry:
    iteration: int
    rank: int
    fit_rel_error: float
    beta: float
    alpha_min: float
    alpha_max: float
    pi_min: float
    pi_max: float


@dataclass
class FitResult:
    coeff_mean: List[float]
    alpha: List[float]
    pi: List[float]
    beta: float
    rank_final: int
    history: List[Dict[str, Any]]
    device: str


class ContinuousBayesianCP:
    """
    CPU-native design with GPU acceleration for heavy tensor ops.

    Core method:
    - Continuous CP factors
    - Gaussian slab over coefficients
    - Soft spike-and-slab-style support shrinkage
    - No unrestricted forward growth
    """

    def __init__(
        self,
        rank_init: int,
        max_iters: int = 80,
        pi_prior: float = 0.40,
        logit_gain: float = 5.0,
        drop_prob_thresh: float = 0.08,
        patience: int = 6,
        merge_corr: float = 0.997,
        a0: float = 1e-2,
        b0: float = 1e-2,
        c0: float = 1e-2,
        d0: float = 1e-2,
        dtype: torch.dtype = torch.float32,
        device: str = "auto",
        seed: int = 0,
    ):
        self.rank_init = rank_init
        self.max_iters = max_iters
        self.pi_prior = pi_prior
        self.logit_gain = logit_gain
        self.drop_prob_thresh = drop_prob_thresh
        self.patience = patience
        self.merge_corr = merge_corr
        self.a0 = a0
        self.b0 = b0
        self.c0 = c0
        self.d0 = d0
        self.dtype = dtype
        self.device = self._resolve_device(device)
        self.seed = seed
        self._factors: Optional[List[List[torch.Tensor]]] = None
        self._coeff_mean: Optional[torch.Tensor] = None
        self._alpha: Optional[torch.Tensor] = None
        self._pi: Optional[torch.Tensor] = None
        self._beta: Optional[torch.Tensor] = None
        self._history: List[FitHistoryEntry] = []

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _rng(self) -> torch.Generator:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        return g

    def _normalize(self, v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return v / torch.clamp(torch.linalg.norm(v), min=eps)

    def _rank1_tensor(self, factors: List[torch.Tensor]) -> torch.Tensor:
        out = factors[0]
        for f in factors[1:]:
            out = torch.einsum("...i,j->...ij", out.reshape(*out.shape, 1), f).reshape(*out.shape, f.numel())
            # reshape progressively to correct higher-order tensor shape
            out = out.reshape(*[x.numel() for x in factors[: factors.index(f)+1]])
        # safer rebuild:
        out = factors[0]
        for f in factors[1:]:
            out = torch.einsum("...i,j->...ij", out.unsqueeze(-1), f).reshape(*out.shape, f.numel())
        return out

    def _rank1_tensor_fast(self, factors: List[torch.Tensor]) -> torch.Tensor:
        out = factors[0]
        for f in factors[1:]:
            out = torch.outer(out.reshape(-1), f).reshape(*out.shape, f.numel())
        return out

    def _cp_reconstruct(self, lam: torch.Tensor, factors_list: List[List[torch.Tensor]]) -> torch.Tensor:
        shape = [f.numel() for f in factors_list[0]]
        X = torch.zeros(shape, dtype=self.dtype, device=self.device)
        for k in range(len(factors_list)):
            X = X + lam[k] * self._rank1_tensor_fast(factors_list[k])
        return X

    def _contract_except(self, X: torch.Tensor, factors: List[torch.Tensor], mode: int) -> torch.Tensor:
        res = X
        modes = [m for m in range(X.ndim) if m != mode]
        for m in sorted(modes, reverse=True):
            res = torch.tensordot(res, factors[m], dims=([m], [0]))
        return res

    def _component_corr(self, fa: List[torch.Tensor], fb: List[torch.Tensor]) -> float:
        prod = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        for a, b in zip(fa, fb):
            prod = prod * torch.dot(a, b)
        return float(prod.abs().item())

    def _gram_from_factors(self, factors_list: List[List[torch.Tensor]]) -> torch.Tensor:
        K = len(factors_list)
        G = torch.ones((K, K), dtype=self.dtype, device=self.device)
        for i in range(K):
            for j in range(i, K):
                val = torch.tensor(1.0, dtype=self.dtype, device=self.device)
                for m in range(len(factors_list[i])):
                    val = val * torch.dot(factors_list[i][m], factors_list[j][m])
                G[i, j] = val
                G[j, i] = val
        return G

    def _b_from_data(self, Y: torch.Tensor, factors_list: List[List[torch.Tensor]]) -> torch.Tensor:
        vals = []
        for fs in factors_list:
            vals.append(torch.sum(Y * self._rank1_tensor_fast(fs)))
        return torch.stack(vals) if vals else torch.empty(0, dtype=self.dtype, device=self.device)

    def _refit_current(self, Y: torch.Tensor, factors_list: List[List[torch.Tensor]], alpha: torch.Tensor, beta: torch.Tensor):
        K = len(factors_list)
        if K == 0:
            zmat = torch.empty((0, 0), dtype=self.dtype, device=self.device)
            zvec = torch.empty((0,), dtype=self.dtype, device=self.device)
            return zmat, zvec, zmat, zvec
        G = self._gram_from_factors(factors_list)
        b = self._b_from_data(Y, factors_list)
        Prec = beta * G + torch.diag(alpha)
        S = torch.linalg.inv(Prec + 1e-8 * torch.eye(K, dtype=self.dtype, device=self.device))
        m = beta * (S @ b)
        return G, b, S, m

    def _vb_score(self, G, b, y_norm2, beta, alpha, m, S):
        p = b.numel()
        if p == 0:
            return float((-0.5 * beta * y_norm2).item())
        trGS = torch.trace(G @ S)
        resid_quad = y_norm2 - 2 * torch.dot(b, m) + torch.dot(m, G @ m) + trGS
        logdet_prec = torch.linalg.slogdet(beta * G + torch.diag(alpha) + 1e-8 * torch.eye(p, dtype=self.dtype, device=self.device))[1]
        logdet_S = torch.linalg.slogdet(S + 1e-8 * torch.eye(p, dtype=self.dtype, device=self.device))[1]
        sparsity_pen = torch.sum(torch.log1p(alpha))
        score = -0.5 * beta * resid_quad + 0.5 * logdet_prec + 0.5 * logdet_S - 0.5 * sparsity_pen
        return float(score.item())

    def _init_factors(self, shape: torch.Size) -> List[List[torch.Tensor]]:
        g = self._rng()
        factors = []
        for _ in range(self.rank_init):
            fs = []
            for s in shape:
                v = torch.randn((s,), generator=g, dtype=self.dtype)
                v = self._normalize(v).to(self.device)
                fs.append(v)
            factors.append(fs)
        return factors

    def fit(self, Y_in) -> FitResult:
        Y = torch.as_tensor(Y_in, dtype=self.dtype, device=self.device)
        y_norm2 = torch.sum(Y * Y)
        factors_list = self._init_factors(Y.shape)
        alpha = torch.ones((self.rank_init,), dtype=self.dtype, device=self.device)
        beta = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        pi = torch.full((self.rank_init,), self.pi_prior, dtype=self.dtype, device=self.device)
        low_count = torch.zeros((self.rank_init,), dtype=torch.int64, device=self.device)
        history: List[FitHistoryEntry] = []

        for it in range(self.max_iters):
            K = len(factors_list)
            if K == 0:
                break

            G, b, S, m = self._refit_current(Y, factors_list, alpha, beta)
            Xhat = self._cp_reconstruct(m, factors_list)

            for k in range(K):
                if float(torch.abs(m[k]).item()) < 1e-12:
                    continue
                Rk = Y - (Xhat - m[k] * self._rank1_tensor_fast(factors_list[k]))
                for mode in range(Y.ndim):
                    factors_list[k][mode] = self._normalize(self._contract_except(Rk, factors_list[k], mode))
                Xhat = self._cp_reconstruct(m, factors_list)

            G, b, S, m = self._refit_current(Y, factors_list, alpha, beta)

            second = m**2 + torch.diag(S)
            alpha = (self.a0 + 0.5) / (self.b0 + 0.5 * second)

            trGS = torch.trace(G @ S)
            resid_quad = y_norm2 - 2 * torch.dot(b, m) + torch.dot(m, G @ m) + trGS
            beta = (self.c0 + 0.5 * Y.numel()) / (self.d0 + 0.5 * resid_quad + 1e-12)

            # soft spike-and-slab shrinkage
            med_energy = torch.median(second) + 1e-12
            z_energy = torch.log(second / med_energy)
            z_alpha = torch.log((torch.median(alpha) + 1e-12) / (alpha + 1e-12))
            logits = math.log(self.pi_prior / (1.0 - self.pi_prior)) + self.logit_gain * (0.7 * z_energy + 0.3 * z_alpha)
            pi = torch.sigmoid(logits)
            low_count = torch.where(pi < self.drop_prob_thresh, low_count + 1, torch.zeros_like(low_count))

            # patient pruning
            keep = [k for k in range(len(factors_list)) if not (float(pi[k].item()) < self.drop_prob_thresh and int(low_count[k].item()) >= self.patience)]
            if len(keep) == 0:
                keep = [int(torch.argmax(second).item())]
            if len(keep) < len(factors_list):
                factors_list = [factors_list[k] for k in keep]
                alpha = alpha[keep]
                pi = pi[keep]
                low_count = low_count[keep]

            # conservative duplicate merge
            merged = True
            while merged and len(factors_list) > 1:
                merged = False
                G, b, S, m = self._refit_current(Y, factors_list, alpha, beta)
                energy = m**2 + torch.diag(S)
                base_score = self._vb_score(G, b, y_norm2, beta, alpha, m, S)
                best_drop = None
                best_gain = 0.0
                for i in range(len(factors_list)):
                    for j in range(i + 1, len(factors_list)):
                        if self._component_corr(factors_list[i], factors_list[j]) > self.merge_corr:
                            drop = i if float(energy[i].item()) < float(energy[j].item()) else j
                            trial_factors = [factors_list[t] for t in range(len(factors_list)) if t != drop]
                            trial_alpha = torch.stack([alpha[t] for t in range(len(alpha)) if t != drop])
                            Gt, bt, St, mt = self._refit_current(Y, trial_factors, trial_alpha, beta)
                            trial_score = self._vb_score(Gt, bt, y_norm2, beta, trial_alpha, mt, St)
                            gain = trial_score - base_score
                            if gain > best_gain + 1e-8:
                                best_gain = gain
                                best_drop = drop
                if best_drop is not None:
                    factors_list = [factors_list[t] for t in range(len(factors_list)) if t != best_drop]
                    alpha = torch.stack([alpha[t] for t in range(len(alpha)) if t != best_drop])
                    pi = torch.stack([pi[t] for t in range(len(pi)) if t != best_drop])
                    low_count = torch.stack([low_count[t] for t in range(len(low_count)) if t != best_drop])
                    merged = True

            G, b, S, m = self._refit_current(Y, factors_list, alpha, beta)
            Xhat = self._cp_reconstruct(m, factors_list)
            fit_rel_error = float((torch.linalg.norm(Xhat - Y) / (torch.linalg.norm(Y) + 1e-12)).item())
            history.append(
                FitHistoryEntry(
                    iteration=it + 1,
                    rank=len(factors_list),
                    fit_rel_error=fit_rel_error,
                    beta=float(beta.item()),
                    alpha_min=float(torch.min(alpha).item()),
                    alpha_max=float(torch.max(alpha).item()),
                    pi_min=float(torch.min(pi).item()),
                    pi_max=float(torch.max(pi).item()),
                )
            )

        self._factors = factors_list
        G, b, S, m = self._refit_current(Y, factors_list, alpha, beta)
        self._coeff_mean = m.detach().cpu()
        self._alpha = alpha.detach().cpu()
        self._pi = pi.detach().cpu()
        self._beta = beta.detach().cpu()
        self._history = history

        return FitResult(
            coeff_mean=self._coeff_mean.tolist(),
            alpha=self._alpha.tolist(),
            pi=self._pi.tolist(),
            beta=float(self._beta.item()),
            rank_final=len(factors_list),
            history=[asdict(h) for h in history],
            device=self.device,
        )

    def reconstruct(self) -> torch.Tensor:
        if self._factors is None or self._coeff_mean is None:
            raise RuntimeError("Call fit() first.")
        coeff = self._coeff_mean.to(self.device, dtype=self.dtype)
        return self._cp_reconstruct(coeff, self._factors)


def generate_synthetic_cp(shape, rank, noise_std=0.0, seed=0):
    model = ContinuousBayesianCP(rank_init=rank, max_iters=1, device="cpu", seed=seed)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    coeff = torch.empty((rank,), dtype=torch.float32).uniform_(0.8, 1.2, generator=rng)
    factors = []
    for _ in range(rank):
        fs = []
        for s in shape:
            v = torch.randn((s,), generator=rng, dtype=torch.float32)
            fs.append(v / torch.clamp(torch.linalg.norm(v), min=1e-12))
        factors.append(fs)
    X = model._cp_reconstruct(coeff, [[f.to("cpu") for f in fs] for fs in factors]).cpu()
    if noise_std > 0:
        Xn = X + noise_std * torch.randn_like(X, generator=rng)
    else:
        Xn = X.clone()
    return X, Xn


if __name__ == "__main__":
    shape = (6, 5, 4)
    true_rank = 3
    init_rank = 5
    Xtrue, Y = generate_synthetic_cp(shape, true_rank, noise_std=0.01, seed=7)
    model = ContinuousBayesianCP(rank_init=init_rank, max_iters=40, device="auto", seed=17)
    result = model.fit(Y)
    Xhat = model.reconstruct().detach().cpu()
    rel_true = float((torch.linalg.norm(Xhat - Xtrue) / (torch.linalg.norm(Xtrue) + 1e-12)).item())
    rel_obs = float((torch.linalg.norm(Xhat - Y) / (torch.linalg.norm(Y) + 1e-12)).item())
    print("device:", result.device)
    print("rank_final:", result.rank_final)
    print("rel_error_to_true:", rel_true)
    print("rel_error_to_observed:", rel_obs)
