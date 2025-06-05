import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FiringRateRNN(nn.Module):
    """
    Continuous-time firing-rate RNN:
      τ dx/dt = -x + W_rec @ r + W_in @ u + b_rec + noise
      r = tanh(x)
      z = W_out^T @ r + b_out
    where：
      - x: (batch, N) hidden state (pre-activation)
      - r: (batch, N) firing rate
      - u: (batch, T, input_dim) input
      - z: (batch, T) or (batch, 1) network output
    """
    def __init__(
        self,
        hidden_size: int = 200,
        tau: float = 10.0,
        dt: float = 1.0,
        input_dim: int = 2,
        noise_std: float = 0.01
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tau = tau
        self.dt = dt
        self.noise_std = noise_std

        # J ~ N(0, 1/N) recurrent weight
        self.W_rec = nn.Parameter(
            torch.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        )
        # B ~ U(-1,1) input weight
        self.W_in = nn.Parameter(
            torch.empty(hidden_size, input_dim).uniform_(-1.0, 1.0)
        )
        # recurrent bias
        self.b_rec = nn.Parameter(torch.zeros(hidden_size))
        # output layer
        self.W_out = nn.Parameter(torch.zeros(hidden_size, 1))
        self.b_out = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, x0: torch.Tensor = None, return_all: bool = True):
        """
        Parameters:
            u (torch.Tensor): Input sequence of shape (batch, T, input_dim).
            x0 (torch.Tensor, optional): Initial hidden state of shape (batch, hidden_size).
                                         Defaults to a zero tensor if not provided.
            return_all (bool): If True, returns the output at every time step;
                               if False, returns only the final time step output.

        Returns:
            torch.Tensor: 
                - If return_all=True, a tensor of shape (batch, T) containing all outputs.
                - Otherwise, a tensor of shape (batch, 1) containing only the final output.
        """

        batch, T, _ = u.shape
        # intialize hidden state x0
        x = x0 if (x0 is not None) else torch.zeros(batch, self.hidden_size, device=u.device)

        outputs = []
        for t in range(T):
            r = torch.tanh(x)                                            # firing rate
            z_t = (r @ self.W_out).squeeze(-1) + self.b_out              # (batch,)

            if return_all:
                outputs.append(z_t.unsqueeze(1))                        # (batch,1)

            # Euler integration of continuous-time dynamics
            noise = torch.randn_like(x) * self.noise_std
            dx = (
                -x
                + r @ self.W_rec.T
                + u[:,t,:] @ self.W_in.T
                + self.b_rec
                + noise
            ) * (self.dt / self.tau)
            x = x + dx

        if return_all:
            return torch.cat(outputs, dim=1)    # (batch, T)
        else:
            return z_t.unsqueeze(-1)            # (batch, 1)