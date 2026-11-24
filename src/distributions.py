import numpy as np


def eight_gaussians(n_samples, radius=4.0, std=0.5, seed=None):
    """Sample equally weighted Gaussian clusters on a circle (8 modes)."""
    if n_samples < 1 or radius <= 0 or std <= 0:
        raise ValueError("n_samples, radius, and std must be positive")
    rng = np.random.default_rng(seed)
    angles = 2 * np.pi * np.arange(8) / 8
    centers = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    labels = rng.integers(0, 8, size=n_samples)
    return centers[labels] + rng.normal(0, std, size=(n_samples, 2))


def spiral(n_samples, turns=2.0, radius=4.0, noise=0.15, seed=None):
    """Sample one noisy Archimedean spiral (configurable toy distribution).

    The precise spiral in the assignment figure is unspecified here;
    `turns`, `radius`, and `noise` can be adjusted to match it.
    """
    if n_samples < 1 or turns <= 0 or radius <= 0 or noise < 0:
        raise ValueError("invalid spiral parameters")
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n_samples)
    theta = 2 * np.pi * turns * u
    r = radius * u
    x = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    return x + rng.normal(0, noise, size=(n_samples, 2))


def potential(x, eps=3.0, c=1.0, d=0.5):
    """V(x0,x1) = eps*[c*(x0**2-1)**2 + (x0-x1)**2 + d*(x0+x1)].

    Accepts a point of shape (2,) or an array with last dimension 2.
    """
    x = np.asarray(x)
    if x.shape[-1] != 2:
        raise ValueError("x must have last dimension of size 2")
    x0, x1 = x[..., 0], x[..., 1]
    return eps * (c * (x0**2 - 1)**2 + (x0 - x1)**2 + d * (x0 + x1))


def metropolis(n_samples, kBT=1.0, step_size=0.5, burn_in=5000, seed=None, initial=(0.0, 0.0)):
    """Random-walk Metropolis sampler for the 2D Boltzmann distribution.

    Returns (samples, acceptance_rate). Consecutive samples are correlated;
    burn_in does not guarantee equilibration, especially across wells.
    """
    if n_samples < 1 or kBT <= 0 or step_size <= 0 or burn_in < 0:
        raise ValueError("n_samples, kBT, step_size must be positive; burn_in >= 0")
    rng = np.random.default_rng(seed)
    x = np.asarray(initial, dtype=float).copy()
    if x.shape != (2,):
        raise ValueError("initial must contain exactly two coordinates")
    samples = np.empty((n_samples, 2), dtype=float)
    accepted = 0
    total_steps = n_samples + burn_in

    for i in range(total_steps):
        proposal = x + rng.normal(0, step_size, size=2)
        delta_V = potential(proposal) - potential(x)
        # Log form prevents overflow of exp for strongly downhill proposals.
        if delta_V <= 0 or np.log(rng.random()) < -delta_V / kBT:
            x = proposal
            accepted += 1
        if i >= burn_in:
            samples[i - burn_in] = x

    return samples, accepted / total_steps