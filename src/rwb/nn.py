"""Global MLP forecaster trained on GPU; samples are gathered from on-device panel tensors.

One network predicts all 8 horizons for a (route, origin) pair from:
recent history of y, half-hour shipments x and statuses; the same clock slots over the past
two weeks; and embeddings of route, origin slot and weekday. Loss is batch WAPE, which is the
metric's first term, so large routes weigh exactly as in the score.
"""

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rwb.data import HORIZON, Panel
from rwb.features import DAY, MAX_LOOKBACK, WEEK, Cumsums

log = logging.getLogger(__name__)


@dataclass
class NNConfig:
    days: int = 42
    recent: int = 96  # steps of raw history per channel
    slot_days: int = 14  # past days of same-slot values around the target window
    hidden: int = 512
    layers: int = 3
    dropout: float = 0.1
    emb_route: int = 16
    lr: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 6
    batch: int = 4096
    # "mlp" flattens the recent history; "gru" reads it as a sequence before the same MLP head.
    arch: str = "mlp"
    rnn_hidden: int = 128
    seed: int = 0


class Data:
    """Panel tensors on device plus a gather function building model inputs."""

    def __init__(self, panel: Panel, cums: Cumsums, device: str) -> None:
        self.device = device
        y = torch.tensor(np.nan_to_num(panel.y), dtype=torch.float32, device=device)
        x = torch.tensor(cums.x, dtype=torch.float32, device=device)
        s = torch.tensor(np.nan_to_num(panel.s), dtype=torch.float32, device=device)
        self.ch = torch.cat([y[None], x[None], s], dim=0)  # (8, R, T)
        cy = torch.tensor(cums.y, dtype=torch.float64, device=device)
        # Route scale per time: trailing weekly mean (+1), exactly as in the tree features.
        self.scale = ((cy[:, WEEK:] - cy[:, :-WEEK]) / WEEK + 1.0).float()  # index o -> column o - WEEK + 1
        cs = torch.tensor(cums.s, dtype=torch.float64, device=device)
        self.s_scale = ((cs[:, :, WEEK:] - cs[:, :, :-WEEK]) / WEEK + 1.0).float()
        ts = panel.times
        self.slot = torch.tensor((ts.hour * 2 + ts.minute // 30).to_numpy(), device=device)
        self.dow = torch.tensor(ts.day_of_week.to_numpy(), device=device)
        self.R = panel.n_routes

    def _scale_at(self, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        return self.scale[r, o - WEEK + 1]

    def inputs(self, r: torch.Tensor, o: torch.Tensor, cfg: NNConfig) -> tuple[torch.Tensor, ...]:
        """Returns (seq (B, L, 8) oldest first, static (B, S), route, slot, dow, scale)."""
        sc = self._scale_at(r, o)[:, None]
        k = torch.arange(cfg.recent, device=self.device)
        idx = (o[:, None] - k[None, :])  # (B, L)
        yx = self.ch[:2][:, r[:, None], idx] / sc[None]  # (2, B, L)
        ss = self.ch[2:][:, r[:, None], idx] / self.s_scale[:, r, o - WEEK + 1][:, :, None]  # (6, B, L)
        seq = torch.flip(torch.cat([yx, ss], dim=0).permute(1, 2, 0), dims=[1])
        # Same clock slots as the target window (and 4 steps before it) on previous days.
        j = torch.arange(-4, HORIZON + 1, device=self.device)
        d = torch.arange(1, cfg.slot_days + 1, device=self.device)
        sidx = (o[:, None, None] + j[None, :, None] - DAY * d[None, None, :]).reshape(len(o), -1)
        slots = self.ch[:2][:, r[:, None], sidx] / sc[None]  # (2, B, J*D)
        wk = torch.arange(1, 5, device=self.device)
        widx = (o[:, None, None] + j[None, :, None] - WEEK * wk[None, None, :]).reshape(len(o), -1)
        weeks = self.ch[:1][:, r[:, None], widx] / sc[None]
        static = torch.cat([slots.permute(1, 0, 2).reshape(len(o), -1), weeks.permute(1, 0, 2).reshape(len(o), -1),
                            torch.log(sc)], dim=1)
        return torch.clamp(seq, 0, 20), torch.clamp(static, 0, 20), r, self.slot[o], self.dow[o], sc[:, 0]

    def base(self, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        """Seasonal reference (B, H) in units of scale, same as the tree baseline."""
        h = torch.arange(1, HORIZON + 1, device=self.device)
        t = o[:, None] + h[None, :]
        y = self.ch[0]
        slot7 = torch.stack([y[r[:, None], t - k * DAY] for k in range(1, 8)]).mean(0)
        rm48 = torch.stack([y[r, o - k] for k in range(DAY)]).mean(0)[:, None]
        return (0.5 * rm48 + 0.5 * slot7) / self._scale_at(r, o)[:, None]

    def target(self, r: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        h = torch.arange(1, HORIZON + 1, device=self.device)
        return self.ch[0][r[:, None], o[:, None] + h[None, :]]


class Net(nn.Module):
    def __init__(self, seq_shape: tuple[int, int], n_static: int, n_routes: int, cfg: NNConfig) -> None:
        super().__init__()
        self.er = nn.Embedding(n_routes, cfg.emb_route)
        self.es = nn.Embedding(48, 8)
        self.ed = nn.Embedding(7, 4)
        seq_len, n_ch = seq_shape
        self.rnn: nn.GRU | None = None
        if cfg.arch == "gru":
            self.rnn = nn.GRU(n_ch, cfg.rnn_hidden, batch_first=True)
            n_seq = cfg.rnn_hidden
        elif cfg.arch == "mlp":
            n_seq = seq_len * n_ch
        else:
            raise ValueError(cfg.arch)
        dims = [n_seq + n_static + cfg.emb_route + 12] + [cfg.hidden] * cfg.layers
        blocks: list[nn.Module] = []
        for a, b in itertools.pairwise(dims):
            blocks += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU(), nn.Dropout(cfg.dropout)]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(cfg.hidden, HORIZON)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, seq: torch.Tensor, static: torch.Tensor, r: torch.Tensor, slot: torch.Tensor,
                dow: torch.Tensor) -> torch.Tensor:
        s = self.rnn(seq)[1][-1] if self.rnn is not None else seq.flatten(1)
        z = torch.cat([s, static, self.er(r), self.es(slot), self.ed(dow)], dim=1)
        return self.head(self.body(z))


def nn_fit_predict(panel: Panel, cums: Cumsums, label_cutoff: int, pred_origins: np.ndarray,
                   cfg: NNConfig) -> np.ndarray:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = Data(panel, cums, dev)
    R = panel.n_routes
    hi = label_cutoff - HORIZON  # all 8 labels must be known
    lo = max(MAX_LOOKBACK, hi - cfg.days * DAY)
    origins = np.arange(lo, hi + 1)
    pairs_r = np.tile(np.arange(R), len(origins))
    pairs_o = np.repeat(origins, R)
    n = len(pairs_o)

    seq0, static0 = data.inputs(torch.zeros(2, dtype=torch.long, device=dev), torch.full((2,), hi, device=dev), cfg)[:2]
    net = Net(tuple(seq0.shape[1:]), static0.shape[1], R, cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * (n // cfg.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.1)
    pr = torch.tensor(pairs_r, device=dev)
    po = torch.tensor(pairs_o, device=dev)
    net.train()
    step = 0
    for ep in range(cfg.epochs):
        perm = torch.tensor(rng.permutation(n), device=dev)
        tot, cnt = 0.0, 0.0
        for b in range(n // cfg.batch):
            ib = perm[b * cfg.batch : (b + 1) * cfg.batch]
            r, o = pr[ib], po[ib]
            seq, static, rr, sl, dw, sc = data.inputs(r, o, cfg)
            pred = (data.base(r, o) + net(seq, static, rr, sl, dw)) * sc[:, None]
            y = data.target(r, o)
            loss = (pred - y).abs().sum() / y.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            tot += float(loss) * float(y.sum())
            cnt += float(y.sum())
        log.info("epoch %d train wape %.4f", ep, tot / cnt)

    net.eval()
    out = np.empty((len(pred_origins), R, HORIZON))
    with torch.no_grad():
        for i, o_ in enumerate(pred_origins):
            r = torch.arange(R, device=dev)
            o = torch.full((R,), int(o_), device=dev)
            seq, static, rr, sl, dw, sc = data.inputs(r, o, cfg)
            pred = (data.base(r, o) + net(seq, static, rr, sl, dw)) * sc[:, None]
            out[i] = pred.clamp_min(0).cpu().numpy()
    return out
