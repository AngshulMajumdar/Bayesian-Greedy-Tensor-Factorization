import torch
from continuous_bayesian_cp import ContinuousBayesianCP


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
    Y = X + noise_std * torch.randn(X.shape, generator=rng, dtype=X.dtype) if noise_std > 0 else X.clone()
    return X, Y


def test_smoke_fit_runs():
    Xtrue, Y = generate_synthetic_cp((6, 5, 4), rank=3, noise_std=0.01, seed=7)
    model = ContinuousBayesianCP(rank_init=5, max_iters=20, device="cpu", seed=17)
    result = model.fit(Y)
    Xhat = model.reconstruct().detach().cpu()
    assert result.rank_final >= 1
    assert Xhat.shape == Y.shape
    rel_obs = float((torch.linalg.norm(Xhat - Y) / (torch.linalg.norm(Y) + 1e-12)).item())
    assert rel_obs < 0.25
