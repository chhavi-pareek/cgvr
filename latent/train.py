import numpy as np
import torch
import torch.nn as nn

from latent.corpus import ANIMATION, BEHAVIOUR, GROUPS, load
from latent.embed import CACHE, embed

D = 16
N_TRAIN = 4000
SPLIT_SEED = 123


class AE(nn.Module):
    def __init__(self, d_in, d_lat=D, h=128, hh=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, d_lat))
        self.rec = nn.Sequential(nn.Linear(d_lat, h), nn.Tanh(), nn.Linear(h, d_in))
        self.heads = nn.ModuleDict(
            {k: nn.Sequential(nn.Linear(d_lat, hh), nn.Tanh(), nn.Linear(hh, d)) for k, d in GROUPS.items()}
        )

    def decode(self, z):
        out = {k: h(z) for k, h in self.heads.items()}
        out["gesture"] = torch.softmax(out["gesture"], -1)
        return out


def get_data():
    texts, tg, _ = load()
    x = torch.from_numpy(embed(texts))
    y = {k: torch.from_numpy(v) for k, v in tg.items()}
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(x))
    return x, y, perm[:N_TRAIN], perm[N_TRAIN:]


def head_var(y, idx, names):
    return sum(y[k][idx].var(0).sum() for k in names)


def head_err(pred, y, idx, names, var):
    """Normalised MSE: 1.0 = predicting the training mean, 0.0 = exact."""
    return sum(((pred[k] - y[k][idx]) ** 2).sum(1).mean() for k in names) / var


def train(seed, nested, epochs=200, bs=256, lr=2e-3):
    torch.manual_seed(seed)
    x, y, tr, _ = get_data()
    m = AE(x.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    xv = x[tr].var(0).sum()
    gv = {k: y[k][tr].var(0).sum() for k in GROUPS}
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        p = torch.from_numpy(tr)[torch.randperm(len(tr), generator=g)]
        for b in p.split(bs):
            z = m.enc(x[b])
            if nested:
                k = torch.randint(1, D + 1, (len(b),), generator=g)
                z = z * (torch.arange(D)[None] < k[:, None])
            pred = m.decode(z)
            loss = ((m.rec(z) - x[b]) ** 2).sum(1).mean() / xv
            loss = loss + sum(((pred[k_] - y[k_][b]) ** 2).sum(1).mean() / gv[k_] for k_ in GROUPS)
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        z = m.enc(x[tr])
    if nested:
        order, fill = torch.arange(D), torch.zeros(D)
    else:
        order, fill = torch.argsort(z.var(0), descending=True), z.mean(0)
    CACHE.mkdir(exist_ok=True)
    torch.save({"state": m.state_dict(), "order": order, "fill": fill}, CACHE / f"ae_{'nested' if nested else 'plain'}_s{seed}.pt")
    return m, order, fill


def load_model(seed, nested):
    x = get_data()[0]
    ck = torch.load(CACHE / f"ae_{'nested' if nested else 'plain'}_s{seed}.pt")
    m = AE(x.shape[1])
    m.load_state_dict(ck["state"])
    return m.eval(), ck["order"], ck["fill"]


if __name__ == "__main__":
    x, y, tr, te = get_data()
    for nested in (True, False):
        for seed in range(3):
            m, order, fill = train(seed, nested)
            with torch.no_grad():
                pred = m.decode(m.enc(x[te]))
            b = head_err(pred, y, te, BEHAVIOUR, head_var(y, tr, BEHAVIOUR)).item()
            a = head_err(pred, y, te, ANIMATION, head_var(y, tr, ANIMATION)).item()
            print(f"nested={nested} seed={seed} k=16 heldout behaviour={b:.4f} animation={a:.4f}")
