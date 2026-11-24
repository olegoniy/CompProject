# Stochastic Interpolants

Learning probability flows from a standard 2D Gaussian to non-Gaussian target distributions using neural networks.

## Setup

```bash
pip install numpy matplotlib torch jupyter
jupyter notebook notebooks/experiments.ipynb
```

Make sure the notebook can import modules from `src/` (for example, add `../src` to `sys.path` when the working directory is `notebooks/`).

## Experiments

- Compare generated and target samples.
- Visualize training loss, velocity fields, and trajectories.
- Study sensitivity to RK4 steps, temperature, and training-set size.

## References

- Albergo & Vanden-Eijnden (2023), *Building Normalizing Flows with Stochastic Interpolants*.
- Albergo, Boffi & Vanden-Eijnden (2025), [*Stochastic Interpolants: A Unifying Framework for Flows and Diffusions*](https://www.jmlr.org/papers/volume26/23-1605/23-1605.pdf).
