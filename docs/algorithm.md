# Algorithm Notes

## Model

We model a tensor using a continuous CP representation:

\[
\mathcal Y pprox \sum_{k=1}^{K} \lambda_k\, u_k^{(1)} \otimes \cdots \otimes u_k^{(d)}
\]

The implementation does **not** use a finite candidate pool. Instead, factors are updated directly.

## Update loop

Each iteration does the following:

1. Update coefficient posterior using the current factor set.
2. Update each factor by residual-style tensor contractions.
3. Update ARD shrinkage parameters.
4. Update noise precision.
5. Compute soft support probabilities.
6. Prune weak components only after persistent low support.
7. Merge near-duplicate components conservatively.

## Why this version is preferred

Three broad families were explored before this codebase was assembled:

- plain continuous VB,
- spike-and-slab-style support rules,
- support correction by evidence-only deletion.

The best balance came from **soft spike-and-slab shrinkage without unrestricted forward growth**.
