"""ML-8 Deep Learning Intro: MLP on MNIST with a manual PyTorch training loop.

Run:  python train.py
Outputs: images/*.png, results/results.json, results/best_model.pt, README.md
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets

import report

SEED = 42
EPOCHS = 12
BATCH_SIZE = 128
VAL_SIZE = 5000
ROOT = Path(__file__).parent
IMG_DIR, RES_DIR = ROOT / "images", ROOT / "results"

# Architecture / hyperparameter variations (everything else is held constant).
CONFIGS = [
    dict(name="baseline", hidden=[128], activation="relu", lr=1e-3),
    dict(name="deeper", hidden=[256, 128], activation="relu", lr=1e-3),
    dict(name="high_lr", hidden=[128], activation="relu", lr=1e-2),
    dict(name="tanh", hidden=[128], activation="tanh", lr=1e-3),
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def describe(cfg):
    layers = " -> ".join(str(h) for h in cfg["hidden"])
    return f"hidden {layers}, {cfg['activation']}, lr={cfg['lr']:g}"


def load_mnist():
    """Return flattened, normalised tensors plus the raw test images for plotting."""
    train = datasets.MNIST(ROOT / "data", train=True, download=True)
    test = datasets.MNIST(ROOT / "data", train=False, download=True)

    def prep(x):
        return ((x.float() / 255.0 - 0.1307) / 0.3081).reshape(len(x), -1)

    X, y = prep(train.data), train.targets
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X), generator=g)
    val_idx, tr_idx = perm[:VAL_SIZE], perm[VAL_SIZE:]
    return (X[tr_idx], y[tr_idx], X[val_idx], y[val_idx],
            prep(test.data), test.targets, test.data.numpy())


def build_model(hidden, activation):
    act = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers, d = [], 28 * 28
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, 10))  # 10 raw class scores (logits)
    return nn.Sequential(*layers)


def train_one_epoch(model, X, y, loss_fn, optimizer):
    model.train()
    order = torch.randperm(len(X))
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), BATCH_SIZE):
        idx = order[i:i + BATCH_SIZE]
        xb, yb = X[idx], y[idx]
        logits = model(xb)              # 1. forward pass
        loss = loss_fn(logits, yb)      # 2. loss
        optimizer.zero_grad()           # 3. clear old gradients
        loss.backward()                 # 4. backward pass (backpropagation)
        optimizer.step()                # 5. update weights
        total_loss += loss.item() * len(idx)
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


@torch.no_grad()
def evaluate(model, X, y, loss_fn):
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(X), 2048):
        logits = model(X[i:i + 2048])
        total_loss += loss_fn(logits, y[i:i + 2048]).item() * len(logits)
        preds.append(logits.argmax(1))
    preds = torch.cat(preds)
    return total_loss / len(X), (preds == y).float().mean().item(), preds


def main():
    IMG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    set_seed()
    Xtr, ytr, Xva, yva, Xte, yte, test_imgs = load_mnist()
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)}")
    loss_fn = nn.CrossEntropyLoss()

    results_cfgs, models = [], {}
    for cfg in CONFIGS:
        set_seed()  # identical seed for every run, so only the config differs
        model = build_model(cfg["hidden"], cfg["activation"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        t0 = time.time()
        for epoch in range(1, EPOCHS + 1):
            tl, ta = train_one_epoch(model, Xtr, ytr, loss_fn, optimizer)
            vl, va, _ = evaluate(model, Xva, yva, loss_fn)
            for k, v in zip(hist, (tl, vl, ta, va)):
                hist[k].append(v)
            print(f"[{cfg['name']}] epoch {epoch:2d}/{EPOCHS} "
                  f"train {tl:.4f}/{ta:.4f} | val {vl:.4f}/{va:.4f}")
        models[cfg["name"]] = model
        results_cfgs.append(dict(
            name=cfg["name"], description=describe(cfg), hidden=cfg["hidden"],
            activation=cfg["activation"], lr=cfg["lr"],
            n_params=sum(p.numel() for p in model.parameters()),
            history=hist, train_loss_final=hist["train_loss"][-1],
            val_loss_final=hist["val_loss"][-1], val_acc_final=hist["val_acc"][-1],
            train_time_s=round(time.time() - t0, 1)))

    # Model selection uses VALIDATION accuracy only. The test set is touched once.
    best = max(results_cfgs, key=lambda c: c["val_acc_final"])
    best_model = models[best["name"]]
    test_loss, test_acc, preds = evaluate(best_model, Xte, yte, loss_fn)
    y_true, y_pred = yte.numpy(), preds.numpy()
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    per_class = (cm.diagonal() / cm.sum(1)).tolist()
    wrong = np.where(y_true != y_pred)[0]
    print(f"\nBEST: {best['name']} | TEST loss {test_loss:.4f} acc {test_acc:.4f}")

    res = dict(seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE,
               n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
               configs=results_cfgs, best=best["name"],
               test=dict(loss=test_loss, acc=test_acc, n_errors=int(len(wrong)),
                         per_class_acc=per_class))
    (RES_DIR / "results.json").write_text(json.dumps(res, indent=2))
    torch.save(best_model.state_dict(), RES_DIR / "best_model.pt")

    report.plot_loss_grid(results_cfgs, IMG_DIR / "loss_curves_all.png")
    report.plot_val_comparison(results_cfgs, IMG_DIR / "val_comparison.png")
    report.plot_best_curves(best, IMG_DIR / "best_model_curves.png")
    report.plot_confusion(cm, IMG_DIR / "confusion_matrix.png")
    report.plot_errors(test_imgs[wrong[:12]], y_true[wrong[:12]], y_pred[wrong[:12]],
                       IMG_DIR / "misclassified_examples.png")
    report.build_readme(res, str(best_model), ROOT / "README.md")
    print("Done. Plots in images/, numbers in results/, README.md regenerated.")


if __name__ == "__main__":
    main()
