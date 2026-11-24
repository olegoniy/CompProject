import torch
import torch.nn as nn

from interpolants import linear_interpolant


def get_device():
    """Use Apple Silicon GPU, CUDA, or CPU, in that order."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VelocityNetwork(nn.Module):
    """Same 3 -> 256 -> 256 -> 256 -> 2 ReLU MLP"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        return self.network(torch.cat([x, t], dim=1))


def train_model(samples, interpolant=linear_interpolant, n_iterations=10000, batch_size=512, lr=1e-3, hidden_dim=256, device=None):
    """Learn v(t, x) from target samples using a chosen stochastic interpolant.

    Args:
        samples: (N, 2) NumPy array or PyTorch tensor from the target distribution.
        interpolant: callable(x0, x1, t) -> (xt, target_velocity).
            For an interpolant with additional parameters, use functools.partial.
        device: torch device or string; autodetected if None.
    Returns:
        (trained_model, list_of_batch_MSE_losses)
    """
    if device is None:
        device = get_device()
    device = torch.device(device)

    target_data = torch.as_tensor(samples, dtype=torch.float32, device=device)
    if target_data.ndim != 2 or target_data.shape[1] != 2 or len(target_data) == 0:
        raise ValueError("samples must have shape (N, 2), with N > 0")
    if n_iterations < 1 or batch_size < 1:
        raise ValueError("n_iterations and batch_size must be positive")

    model = VelocityNetwork(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for _ in range(n_iterations):
        indices = torch.randint(len(target_data), (batch_size,), device=device)
        x1 = target_data[indices]
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1, device=device)
        t = t.clamp(min=1e-5, max=1 - 1e-5)

        xt, target_velocity = interpolant(x0, x1, t)
        predicted_velocity = model(xt, t)
        loss = ((predicted_velocity - target_velocity) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses


@torch.no_grad()
def generate_samples(model, n_samples=5000, n_steps=100, x0=None):
    """Integrate dx/dt = v_theta(x,t) from t=0 to 1 using fixed-step RK4.

    Returns generated samples as a NumPy array (N, 2).
    If x0 is provided, start from those points rather than fresh N(0,I) noise.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be positive")

    device = next(model.parameters()).device
    model.eval()

    if x0 is None:
        if n_samples < 1:
            raise ValueError("n_samples must be positive")
        x = torch.randn(n_samples, 2, device=device)
    else:
        x = torch.as_tensor(x0, dtype=torch.float32, device=device).clone()
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError("x0 must have shape (N, 2)")

    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((len(x), 1), i * dt, device=device)
        k1 = model(x, t)
        k2 = model(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = model(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = model(x + dt * k3, t + dt)
        x = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x.cpu().numpy()

