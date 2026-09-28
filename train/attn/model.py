"""A set-transformer over the candidate tasks of a single turn.

Why attention rather than more features: the agent already scores each task in
isolation and then patches up the interactions by hand -- three features for
distance, crowding and "how many units already chose this op". Those are
stand-ins for a question that is really about the whole set: with these
particular twelve water jobs, two hungry cows and a melon ready to pick, which
one is worth doing? Self-attention lets every candidate see the others, so the
model can represent "this is the only ripe melon" or "there are nine water jobs
and one of them is about to die" without anyone hand-coding those cases.

Columns 8, 12 and 13 of phi (distance, crowding, same-op) are zeroed: they
depend on which unit is being assigned and on what earlier units already took,
and the agent evaluates this network once per turn for the whole set. Those
three terms stay in the linear part, which is still applied per unit, so the
autoregressive behaviour is unchanged.
"""
import torch
import torch.nn as nn

PER_UNIT_COLS = (8, 12, 13)


class TaskSetTransformer(nn.Module):
    def __init__(self, n_feat=26, d_model=48, n_head=4, n_layer=2):
        super().__init__()
        self.inp = nn.Linear(n_feat, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=2 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
            activation='gelu')
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.out = nn.Linear(d_model, 1)
        # Zero output layer: the correction starts at exactly 0, so the agent
        # it ships into is the agent that was measured. Training can only move
        # away from something that already works.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feats, mask):
        """feats (B,N,F), mask (B,N) with 1 for real candidates.

        The output is centred within each turn. The agent chooses by argmax of
        value/(dist+c) * exp(z), so adding the same constant to every candidate
        multiplies all scores equally and changes nothing. A model free to emit
        that constant spends its capacity predicting how rich the farm is --
        measured at a 0.10 ratio of within-turn to between-turn spread, i.e.
        ~90% of the signal was unusable. Centring removes that degree of
        freedom, so every parameter has to earn its place by ranking one
        candidate against another.
        """
        h = self.inp(feats)
        h = self.enc(h, src_key_padding_mask=(mask < 0.5))
        raw = self.out(h).squeeze(-1) * mask
        n = mask.sum(1, keepdim=True).clamp(min=1.0)
        return (raw - (raw.sum(1, keepdim=True) / n)) * mask
