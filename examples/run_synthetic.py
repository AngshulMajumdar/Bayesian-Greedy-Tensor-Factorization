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


def main():
    Xtrue, Y = generate_synthetic_cp((7, 6, 5), rank=3, noise_std=0.01, seed=7)
    model = ContinuousBayesianCP(rank_init=6, max_iters=50, device="auto", seed=17)
    result = model.fit(Y)
    Xhat = model.reconstruct().detach().cpu()
    rel_true = float((torch.linalg.norm(Xhat - Xtrue) / (torch.linalg.norm(Xtrue) + 1e-12)).item())
    rel_obs = float((torch.linalg.norm(Xhat - Y) / (torch.linalg.norm(Y) + 1e-12)).item())
    print("device:", result.device)
    print("rank_final:", result.rank_final)
    print("rel_error_to_true:", rel_true)
    print("rel_error_to_observed:", rel_obs)


if __name__ == "__main__":
    main()
