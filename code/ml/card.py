"""Legacy CARD diffusion components aligned with :math:`y` (performance) and :math:`x` (condition)."""

import torch
from torch.nn.functional import leaky_relu, softplus, relu


class ConditionalLinear(torch.nn.Module):
    """Conditional linear block keyed by diffusion timestep."""
    def __init__(self, dim_in, dim_out, num_steps):
        super(ConditionalLinear, self).__init__()
        self.lin = torch.nn.Linear(dim_in, dim_out)
        self.embed = torch.nn.Embedding(num_steps, dim_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        """Apply the affine transform for step ``t``."""
        return self.embed(t) * self.lin(x)


class FCNN(torch.nn.Module):
    """Fully connected baseline mapping performance :math:`y` to condition :math:`x`."""
    def __init__(self, dim_in, dim_out):
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, dim_out)

    def forward(self, x):
        """Predict conditions :math:`x` for a given performance vector :math:`y`."""
        h = leaky_relu(self.fc1(x))
        h = leaky_relu(self.fc2(h))
        out = self.fc3(h)

        return out

    def fit(self, data_loader, optimizer, loss_func):
        """Train on normalized performance/condition mini-batches."""
        sum_losses = 0

        self.train()
        for x, y in data_loader:
            loss = loss_func(self(x.cuda()), y.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, x):
        """Return CPU predictions using the stored GPU weights."""
        self.eval()
        with torch.no_grad():
            return self(x.cuda()).cpu()


class CARD(torch.nn.Module):
    """Conditional autoregressive diffusion model for generating conditions :math:`x`."""
    def __init__(self, dim_in, dim_out, num_steps, pred_model):
        super(CARD, self).__init__()
        self.num_steps = num_steps
        self.beta = make_beta_schedule(schedule='linear', num_steps=num_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).view(-1, 1).cuda()
        self.pred_model = pred_model

        self.clin1 = ConditionalLinear(dim_in + 2, 128, num_steps)
        self.clin2 = ConditionalLinear(128, 128, num_steps)
        self.clin3 = ConditionalLinear(128, 128, num_steps)
        self.lin = torch.nn.Linear(128, dim_out)

    def forward(self, x, y_t, y_0_hat, t):
        """Predict diffusion noise at timestep ``t`` given performance :math:`y`."""
        _x = torch.cat([x, y_t, y_0_hat], dim=1)
        h = leaky_relu(self.clin1(_x, t))
        h = leaky_relu(self.clin2(h, t))
        h = leaky_relu(self.clin3(h, t))
        out = self.lin(h)

        return out

    def backward(self, x):
        """Reverse diffuse to sample conditions :math:`x` conditioned on performance :math:`y`."""
        y_pred = self.pred_model(x)
        y_t = y_pred
        uncert = None

        for t in reversed(range(0, self.num_steps)):
            uncert = self(x, y_t, y_pred, torch.full((y_t.shape[0],), t).cuda())
            y_0_hat = y_t - (1 - torch.sqrt(self.alpha_bar[t]))
            y_0_hat -= torch.sqrt(1 - self.alpha_bar[t]) * uncert

            if t > 0:
                gamma_0 = self.beta[t] * torch.sqrt(self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
                gamma_1 = ((1 - self.alpha_bar[t - 1]) * torch.sqrt(self.alpha[t])) / (1 - self.alpha_bar[t])
                gamma_2 = (torch.sqrt(self.alpha_bar[t]) - 1) * (torch.sqrt(self.alpha[t]) + torch.sqrt(self.alpha_bar[t - 1]))
                gamma_2 = 1 + (gamma_2 / (1 - self.alpha_bar[t]))
                beta_tilde = ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
                eps = torch.randn_like(y_t).cuda()
                y_t = gamma_0 * y_0_hat + gamma_1 * y_t + gamma_2 * y_pred + beta_tilde * eps
            else:
                y_t = y_0_hat

        return y_t, uncert

    def fit(self, data_loader, optimizer):
        """Optimize CARD on normalized performance/condition data."""
        sum_losses = 0

        self.train()
        for x, y in data_loader:
            x = x.cuda()
            y = y.cuda()

            y_0_hat = self.pred_model(x)
            t = torch.randint(0, self.num_steps, size=(x.shape[0],)).cuda()
            eps = torch.randn_like(y).cuda()

            y_t = q_sample(y, y_0_hat, self.alpha_bar, t, noise=eps)
            eps_pred = self(x, y_t, y_0_hat, t)
            loss = torch.mean((eps - eps_pred)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, x):
        """Return sampled conditions and optionally uncertainty estimates."""
        self.eval()

        with torch.no_grad():
            return self.backward(x.cuda())


def make_beta_schedule(schedule='linear', num_steps=1000, start=1e-5, end=1e-2):
    """Linear beta schedule shared with the main diffusion implementation."""
    if schedule == 'linear':
        return torch.linspace(start, end, num_steps)


def q_sample(y, y_0_hat, alpha_bar, t, noise):
    """Forward diffuse condition tensors (stored in ``y``) with predicted guidance :math:`y_0_hat`."""
    return torch.sqrt(alpha_bar[t]) * y + (1 - torch.sqrt(alpha_bar[t])) * y_0_hat + torch.sqrt(1 - alpha_bar[t]) * noise
