"""Residual policy over the turn's candidate tasks.

The chain problem: farming is plant -> water -> water -> harvest, and one
missed link kills the crop. A policy learned from scratch picks the right task
maybe 10% of the time, so a 20-link chain completes with probability ~1e-20.
An earlier end-to-end network here scored 0-233 against a working agent --
worse than doing nothing, which at least does not waste seeds.

The fix is in the parameterisation, not the training. The policy is

    logit(task) = log(heuristic_score(task)) + scale * f_theta(tasks)

with f_theta's output layer zeroed. At initialisation f_theta = 0 exactly, so
the distribution is the working agent's own scores and every chain completes
from step one. RL then moves the residual, and because it starts on a policy
that already farms, the gradient signal is about *improving* chains rather
than discovering them.

`scale` also bounds how far the network may deviate, so a bad update degrades
the agent gracefully instead of destroying it.
"""
import torch
import torch.nn as nn


class ResidualPolicy(nn.Module):
    def __init__(self, n_feat=26, d_model=32, n_head=4, n_layer=1):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d_model, 1)
        # Critic for the baseline: long-horizon returns are enormous and noisy,
        # so subtracting a learned state value is what makes REINFORCE usable.
        self.value = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Linear(d_model, 1))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, feats, mask, base_logit, scale):
        """feats (B,N,F), mask (B,N), base_logit (B,N) = log heuristic score."""
        h = self.inp(feats)
        h = self.enc(h, src_key_padding_mask=(mask < 0.5))
        r = self.head(h).squeeze(-1)
        # Centre within the turn: adding a constant to every candidate cannot
        # change an argmax or a softmax draw, so an uncentred residual would
        # spend capacity on a term that changes nothing.
        n = mask.sum(1, keepdim=True).clamp(min=1.0)
        r = (r - (r * mask).sum(1, keepdim=True) / n) * mask
        logits = base_logit + scale * r
        logits = logits.masked_fill(mask < 0.5, -1e9)
        # State value from the masked mean of the encoded set.
        pooled = (h * mask.unsqueeze(-1)).sum(1) / n
        return logits, self.value(pooled).squeeze(-1)
