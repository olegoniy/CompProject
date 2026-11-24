import math
import torch


def linear_interpolant(x0, x1, t):
    """Linear interpolation between independently sampled endpoints."""
    xt = (1 - t)*x0 + t*x1
    target_velocity = x1 - x0
    return xt, target_velocity


def trigonometric_interpolant(x0, x1, t):
    """Sine/cosine interpolation, with its exact time derivative."""
    angle = 0.5*math.pi*t
    xt = torch.cos(angle)*x0 + torch.sin(angle)*x1
    target_velocity = 0.5*math.pi*(-torch.sin(angle)*x0 + torch.cos(angle)*x1)
    return xt, target_velocity


def noise_schedule(t, schedule="polynomial", a=1.0, sigmoid_f=8.0):
    """Return (gamma(t), dgamma/dt) before applying noise_scale.

    `sqrt`: sqrt(a*t*(1-t)); its derivative diverges at t=0 and t=1.
    Call it only at interior times. We clamp for numerical protection, not
    to claim a finite mathematical derivative at the endpoints.
    `polynomial`: t*(1-t)
    `sin_squared`: sin(pi*t)**2
    `sigmoid`: compact sigmoid combination from the paper's Table 2.
    `sine`: sin(pi*t), a custom comparison (NOT in Table 2).
    """
    if schedule == "polynomial":
        return t * (1 - t), 1 - 2 * t

    if schedule == "sin_squared":
        return torch.sin(math.pi * t).square(), math.pi * torch.sin(2 * math.pi * t)

    if schedule == "sqrt":
        if a <= 0:
            raise ValueError("a must be positive")
        if torch.any((t <= 0) | (t >= 1)):
            raise ValueError("sqrt noise schedule requires 0 < t < 1 (derivative diverges at endpoints)")
        inside = a * t * (1 - t)
        root = torch.sqrt(inside)
        return root, a * (1 - 2 * t) / (2 * root)

    if schedule == "sigmoid":
        f = sigmoid_f
        if f <= 0:
            raise ValueError("sigmoid_f must be positive")
        # sig(s) = 1/(1+exp(-s)), paper Table 2
        u = f * (t - 0.5)
        s_plus = torch.sigmoid(u + 1)
        s_minus = torch.sigmoid(u - 1)
        offset = torch.sigmoid(t.new_tensor(-f / 2 + 1)) - torch.sigmoid(t.new_tensor(-f / 2 - 1))
        gamma = s_plus - s_minus - offset
        gamma_dot = f * (s_plus * (1 - s_plus) - s_minus * (1 - s_minus))
        return gamma, gamma_dot

    if schedule == "sine":
        return torch.sin(math.pi * t), math.pi * torch.cos(math.pi * t)

    raise ValueError(f"Unknown schedule: {schedule!r}")


def noisy_interpolant(x0, x1, t, noise_scale=0.5, schedule="polynomial", a=1.0,
                      sigmoid_f=8.0, z=None):
    """Linear bridge plus gamma(t)*z, with matching time derivative.

    z is sampled once per training example and reused in BOTH outputs.
    For fair experiments, fix the random seed and vary one schedule at a time.
    """
    if z is None:
        z = torch.randn_like(x0)
    gamma, gamma_dot = noise_schedule(t, schedule=schedule, a=a, sigmoid_f=sigmoid_f)
    xt = (1 - t) * x0 + t * x1 + noise_scale * gamma * z
    velocity = x1 - x0 + noise_scale * gamma_dot * z
    return xt, velocity