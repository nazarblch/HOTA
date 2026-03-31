import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    hidden_layers: tuple
    relu_slope: float = 0.2

    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.hidden_layers):
            x = nn.Dense(dim)(x)
            if i != len(self.hidden_layers) - 1:
                x = nn.leaky_relu(x, negative_slope=self.relu_slope)
        return x

class MLPwTime(nn.Module):
    hidden_layers: tuple
    max_freq: int
    t_emb_dim: int
    relu_slope: float = 0.2

    @nn.compact
    def __call__(self, t, x, _):
        freqs = jnp.arange(1, self.max_freq)
        t_pos = freqs * t
        t_pos = jnp.concatenate([jnp.sin(t_pos) / freqs, jnp.cos(t_pos) / freqs], -1)
        w = nn.Dense(self.t_emb_dim)(t_pos)
        b = nn.Dense(self.t_emb_dim)(t_pos)
        x = nn.Dense(self.t_emb_dim)(x)
        x = w * x + b
        return MLP(self.hidden_layers, self.relu_slope)(x)
