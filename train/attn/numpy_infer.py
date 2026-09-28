"""Pure-numpy forward pass, checked against torch before anything ships.

This is the code that runs inside the submission, so it has to reproduce
torch's pre-norm encoder exactly: LayerNorm, then attention, then residual;
LayerNorm, then GELU feed-forward, then residual. A silent mismatch here would
mean validating one model and submitting another.
"""
import numpy as np


def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _gelu(x):
    """Exact (erf) GELU, matching nn.TransformerEncoderLayer's default.

    The tanh approximation is NOT interchangeable here: it drifted the network
    output by 1e-3, which is a different model from the one that was validated.
    numpy has no erf, so this is Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
    """
    z = x / np.sqrt(2.0)
    sign = np.sign(z)
    a = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (
        -1.453152027 + t * 1.061405429))))
    erf = sign * (1.0 - poly * np.exp(-a * a))
    return 0.5 * x * (1.0 + erf)


def _attn(x, W, layer, n_head):
    n, d = x.shape
    hd = d // n_head
    qkv = x @ W[f'{layer}.self_attn.in_proj_weight'].T + W[f'{layer}.self_attn.in_proj_bias']
    q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
    q = q.reshape(n, n_head, hd).transpose(1, 0, 2)
    k = k.reshape(n, n_head, hd).transpose(1, 0, 2)
    v = v.reshape(n, n_head, hd).transpose(1, 0, 2)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
    s -= s.max(-1, keepdims=True)
    a = np.exp(s)
    a /= a.sum(-1, keepdims=True)
    o = (a @ v).transpose(1, 0, 2).reshape(n, d)
    return o @ W[f'{layer}.self_attn.out_proj.weight'].T + W[f'{layer}.self_attn.out_proj.bias']


def forward(feats, W, n_layer=2, n_head=4):
    """feats (N, F) -> (N,) set-centred score. No padding: N is the real count."""
    h = feats @ W['inp.weight'].T + W['inp.bias']
    for i in range(n_layer):
        p = f'enc.layers.{i}'
        h = h + _attn(_ln(h, W[f'{p}.norm1.weight'], W[f'{p}.norm1.bias']), W, p, n_head)
        y = _ln(h, W[f'{p}.norm2.weight'], W[f'{p}.norm2.bias'])
        y = _gelu(y @ W[f'{p}.linear1.weight'].T + W[f'{p}.linear1.bias'])
        h = h + y @ W[f'{p}.linear2.weight'].T + W[f'{p}.linear2.bias']
    r = (h @ W['out.weight'].T + W['out.bias']).ravel()
    return r - r.mean()
