from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class NeuralScoreTrainingResult:
    model: nn.Module
    train_losses: list[float]
    val_losses: list[float]
    zero_score_baseline_train: float
    zero_score_baseline_val: float
    best_epoch: int
    stopped_early: bool
    device: str
    mean: np.ndarray
    scale: np.ndarray

    @property
    def losses(self):
        return self.train_losses


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return t
        freq = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), half, device=t.device))
        angles = t * freq.view(1, -1)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, t], dim=1)
        return emb


class ScoreNetwork(nn.Module):
    def __init__(self, n_assets: int, hidden_dim: int = 32, time_dim: int = 32, n_hidden_layers: int = 3):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        layers = []
        in_dim = n_assets + time_dim
        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_assets))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embedding(t)], dim=1))


def vp_signal(t: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return torch.exp(-0.5 * beta * t)


def vp_noise_std(t: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    return torch.sqrt(torch.clamp(1.0 - torch.exp(-beta * t), min=1e-8))


def _standardize_train_validation(x_train: np.ndarray, x_val: np.ndarray):
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0, ddof=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x_train - mean) / scale, (x_val - mean) / scale, mean, scale


def _dsm_batch_loss(model, x0, horizon, beta, eps_t):
    bsz = x0.shape[0]
    t = eps_t + (horizon - eps_t) * torch.rand(bsz, 1, device=x0.device)
    noise = torch.randn_like(x0)
    s = vp_signal(t, beta)
    sigma = vp_noise_std(t, beta)
    xt = s * x0 + sigma * noise
    target = -noise / sigma
    pred = model(xt, t)
    loss = ((sigma * (pred - target)) ** 2).sum(dim=1).mean()
    zero_loss = ((sigma * target) ** 2).sum(dim=1).mean()
    return loss, zero_loss


def _evaluate_dsm(model, x, horizon, beta, eps_t, repeats=5):
    model.eval()
    losses, zeros = [], []
    with torch.no_grad():
        for _ in range(max(1, int(repeats))):
            loss, zero = _dsm_batch_loss(model, x, horizon, beta, eps_t)
            losses.append(float(loss.cpu()))
            zeros.append(float(zero.cpu()))
    return float(np.mean(losses)), float(np.mean(zeros))


def train_score_model(
    returns: np.ndarray,
    horizon: float,
    beta: float = 1.0,
    epochs: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_dim: int = 32,
    time_dim: int = 32,
    n_hidden_layers: int = 3,
    validation_fraction: float = 0.20,
    patience: int = 100,
    min_delta: float = 1e-3,
    eval_repeats: int = 5,
    seed: int = 42,
    device: str | None = None,
) -> NeuralScoreTrainingResult:
    """Train a neural score model with chronological validation and early stopping."""
    if horizon <= 0:
        raise ValueError("Neural score training requires T > 0.")
    if not (0.10 <= validation_fraction <= 0.40):
        raise ValueError("validation_fraction must be between 0.10 and 0.40.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.asarray(returns, dtype=np.float32)
    if x.ndim != 2 or len(x) < 20:
        raise ValueError("At least 20 two-dimensional return observations are required.")

    n_val = max(5, int(round(len(x) * validation_fraction)))
    n_train = len(x) - n_val
    if n_train < 10:
        raise ValueError("Not enough observations remain after the validation split.")

    x_train, x_val = x[:n_train], x[n_train:]
    train_z, val_z, mean, scale = _standardize_train_validation(x_train, x_val)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tensor = torch.tensor(train_z, dtype=torch.float32)
    val_tensor = torch.tensor(val_z, dtype=torch.float32, device=device)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=min(int(batch_size), n_train), shuffle=True)

    model = ScoreNetwork(x.shape[1], hidden_dim, time_dim, n_hidden_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    eps_t = max(1e-4, min(1e-2, horizon / 100.0))

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    stopped_early = False

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total, count = 0.0, 0
        for (x0,) in loader:
            x0 = x0.to(device)
            loss, _ = _dsm_batch_loss(model, x0, horizon, beta, eps_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(x0)
            count += len(x0)
        train_loss = total / max(count, 1)
        val_loss, _ = _evaluate_dsm(model, val_tensor, horizon, beta, eps_t, eval_repeats)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - float(min_delta):
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(patience):
            stopped_early = True
            break

    model.load_state_dict(best_state)
    model.eval()

    train_eval = train_tensor.to(device)
    _, zero_train = _evaluate_dsm(model, train_eval, horizon, beta, eps_t, repeats=10)
    _, zero_val = _evaluate_dsm(model, val_tensor, horizon, beta, eps_t, repeats=10)

    return NeuralScoreTrainingResult(
        model=model,
        train_losses=train_losses,
        val_losses=val_losses,
        zero_score_baseline_train=zero_train,
        zero_score_baseline_val=zero_val,
        best_epoch=best_epoch,
        stopped_early=stopped_early,
        device=device,
        mean=mean,
        scale=scale,
    )


@torch.no_grad()
def generate_neural_reverse_diffusion_samples(training_result, m, horizon, beta=1.0, n_steps=200, seed=42):
    if horizon <= 0:
        raise ValueError("Neural reverse diffusion requires T > 0.")
    if m <= 0:
        return np.empty((0, len(training_result.mean)))

    torch.manual_seed(seed)
    model = training_result.model
    model.eval()
    device = training_result.device
    n_assets = len(training_result.mean)
    x = torch.randn(int(m), n_assets, device=device)
    dt = horizon / float(n_steps)

    for k in range(int(n_steps)):
        t_value = horizon - k * dt
        t = torch.full((int(m), 1), max(t_value, 1e-5), device=device)
        score = model(x, t)
        drift = 0.5 * beta * x + beta * score
        x = x + drift * dt + math.sqrt(beta * dt) * torch.randn_like(x)

    z = x.cpu().numpy()
    return z * training_result.scale + training_result.mean


def neural_diffusion_augmented_moments(
    returns,
    m,
    horizon,
    beta=1.0,
    n_steps=200,
    epochs=1000,
    batch_size=32,
    learning_rate=1e-3,
    hidden_dim=32,
    validation_fraction=0.20,
    patience=100,
    min_delta=1e-3,
    seed=42,
):
    from .moments import sample_moments

    real = np.asarray(returns, dtype=float)
    result = train_score_model(
        real,
        horizon=horizon,
        beta=beta,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        validation_fraction=validation_fraction,
        patience=patience,
        min_delta=min_delta,
        seed=seed,
    )
    fake = generate_neural_reverse_diffusion_samples(result, m, horizon, beta, n_steps, seed + 1)
    combined = np.vstack([real, fake]) if len(fake) else real.copy()
    mu_aug, sigma_aug = sample_moments(combined, mle=True)
    return mu_aug, sigma_aug, fake, combined, result
