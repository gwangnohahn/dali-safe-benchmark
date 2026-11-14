"""Neural network modules for mapping performance :math:`y` to conditions :math:`x`.

Code retains historical tensor names: modules receive ``y`` tensors via arguments named
``x`` and emit ``x`` tensors via outputs named ``y``.
"""

import torch
from torch.nn.functional import leaky_relu

# Select compute device automatically for tensor operations.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FCNN(torch.nn.Module):
    """Fully connected regressor predicting conditions :math:`x` from performances :math:`y`."""
    def __init__(self, dim_in, dim_out, ref_dataset):
        super(FCNN, self).__init__()
        self.fc1 = torch.nn.Linear(dim_in, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, dim_out)
        self.ref_dataset = ref_dataset

    def forward(self, x):
        """Return the predicted condition :math:`x` for input performance :math:`y`."""
        h = leaky_relu(self.fc1(x))
        h = leaky_relu(self.fc2(h))
        out = self.fc3(h)

        return out

    def fit(self, data_loader, optimizer, loss_func):
        """Train the network on normalized performance/condition mini-batches."""
        sum_losses = 0

        self.train()
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            loss = loss_func(self(x), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, x):
        """Inference helper mirroring :meth:`forward` on the configured device."""
        self.eval()
        with torch.no_grad():
            return self(x.to(device)).cpu()


class ConditionalLinear(torch.nn.Module):
    """Linear layer conditioned on the diffusion step index."""
    def __init__(self, dim_in, dim_out, num_steps):
        super(ConditionalLinear, self).__init__()
        self.lin = torch.nn.Linear(dim_in, dim_out)
        self.embed = torch.nn.Embedding(num_steps, dim_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        """Apply the conditional affine transform for step ``t``."""
        return self.embed(t) + self.lin(x)


class CondFCNN(torch.nn.Module):
    """Conditional network producing diffusion noise statistics."""
    def __init__(self, dim_in, dim_out, num_steps):
        super(CondFCNN, self).__init__()
        self.clin1 = ConditionalLinear(dim_in, 64, num_steps)
        self.clin2 = ConditionalLinear(64, 64, num_steps)
        self.lin_mean = torch.nn.Sequential(
            torch.nn.Linear(64, 63),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, dim_out)
        )
        self.lin_logvar = torch.nn.Sequential(
            torch.nn.Linear(64, 63),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, dim_out)
        )

    def forward(self, x, t):
        """Sample diffusion noise conditioned on :math:`x` and step ``t``."""
        h = leaky_relu(self.clin1(x, t))
        h = leaky_relu(self.clin2(h, t))
        mean = self.lin_mean(h)
        logvar = self.lin_logvar(h)

        return mean + torch.randn_like(logvar) * torch.exp(0.5 * logvar)


class CondDiffu(torch.nn.Module):
    """Conditional diffusion model generating conditions :math:`x` for target performances :math:`y`."""
    def __init__(self, dim_in, dim_out, num_steps, pred_model, var_stat=None):
        super(CondDiffu, self).__init__()
        self.dim_out = dim_out
        self.num_steps = num_steps
        self.beta = make_beta_schedule(schedule='linear', num_steps=num_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).view(-1, 1).to(device)
        self.pred_model = pred_model
        self.var_stat = var_stat

        self.clin1 = ConditionalLinear(dim_in + 2 * dim_out, 64, num_steps)
        self.clin2 = ConditionalLinear(64, 64, num_steps)
        self.lin = torch.nn.Linear(64, dim_out)

    def forward(self, x, y_t, y_0_hat, t):
        """Predict diffusion noise at step ``t`` conditioned on performance :math:`y`."""
        _x = torch.cat([x, y_t, y_0_hat], dim=1)
        h = leaky_relu(self.clin1(_x, t))
        h = leaky_relu(self.clin2(h, t))
        out = self.lin(h)

        return out

    def backward(self, x):
        """Run the reverse diffusion process to sample conditions :math:`x` given performance :math:`y`."""
        # Alternative predictors for ablation studies:
        # y_pred = self.ref_dataset.get_ref_target(x, train=False).to(device)
        # y_pred = torch.randn((x.shape[0], self.dim_out)).to(device)
        y_pred = self.pred_model(x)
        y_t = y_pred
        _x = x
        # _x = torch.cat([x, y_pred], dim=1)

        for t in reversed(range(0, self.num_steps)):
            y_0_hat = y_t - (1 - torch.sqrt(self.alpha_bar[t]))
            y_0_hat -= torch.sqrt(1 - self.alpha_bar[t]) * self(_x, y_t, y_pred, torch.full((y_t.shape[0],), t).to(device))

            if t > 0:
                gamma_0 = self.beta[t] * torch.sqrt(self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])
                gamma_1 = ((1 - self.alpha_bar[t - 1]) * torch.sqrt(self.alpha[t])) / (1 - self.alpha_bar[t])
                gamma_2 = (torch.sqrt(self.alpha_bar[t]) - 1) * (torch.sqrt(self.alpha[t]) + torch.sqrt(self.alpha_bar[t - 1]))
                gamma_2 = 1 + (gamma_2 / (1 - self.alpha_bar[t]))
                beta_tilde = ((1 - self.alpha_bar[t - 1]) / (1 - self.alpha_bar[t])) * self.beta[t]
                eps = torch.randn_like(y_t).to(device)
                y_t = gamma_0 * y_0_hat + gamma_1 * y_t + gamma_2 * y_pred + beta_tilde * eps
            else:
                y_t = y_0_hat

        return y_t

    def fit(self, data_loader, optimizer):
        """Optimize the diffusion model on normalized performance/condition batches."""
        sum_losses = 0

        self.train()
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            # y_0_hat = self.ref_dataset.get_ref_target(x, train=True).to(device)
            # y_0_hat = torch.randn((x.shape[0], self.dim_out)).to(device)
            y_0_hat = self.pred_model(x)

            t = torch.randint(0, self.num_steps, size=(x.shape[0],)).to(device)
            eps = torch.randn_like(y).to(device)

            y_t = q_sample(y, y_0_hat, self.alpha_bar, t, noise=eps)
            eps_pred = self(x, y_t, y_0_hat, t)
            loss = torch.mean((eps - eps_pred)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sum_losses += loss.item()

        return sum_losses / len(data_loader)

    def predict(self, x):
        """Sample normalized conditions and return them on CPU."""
        self.eval()

        with torch.no_grad():
            return self.backward(x.to(device)).cpu()

    def norm_input_vars(self, x):
        """Normalize raw performance vectors :math:`y`."""
        return (x - self.var_stat['x_mean']) / self.var_stat['x_std']

    def denorm_output_vars(self, y):
        """Denormalize predicted conditions :math:`x`."""
        return self.var_stat['y_std'] * y + self.var_stat['y_mean']


def make_beta_schedule(schedule='linear', num_steps=1000, start=1e-5, end=1e-2):
    """Linear diffusion noise schedule shared across (:math:`x`, :math:`y`) tasks."""
    if schedule == 'linear':
        return torch.linspace(start, end, num_steps)


def q_sample(y, y_0_hat, alpha_bar, t, noise):
    """Forward diffuse a condition tensor (stored in ``y``) with guidance :math:`y_0_hat`."""
    return torch.sqrt(alpha_bar[t]) * y + (1 - torch.sqrt(alpha_bar[t])) * y_0_hat + torch.sqrt(1 - alpha_bar[t]) * noise
