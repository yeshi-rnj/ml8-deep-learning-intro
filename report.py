"""Plots and README generation for ML-8. Needs only numpy + matplotlib (no torch)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2f6fed", "#e8743b", "#2a9d8f", "#9b5de5", "#f0b429"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_loss_grid(configs, path):
    """Train vs validation loss for every configuration (one panel each)."""
    cols = 2
    rows = (len(configs) + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(11, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, c in zip(axes.ravel(), configs):
        ax.axis("on")
        h = c["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color="#2f6fed", lw=2, label="train")
        ax.plot(ep, h["val_loss"], color="#e8743b", lw=2, label="validation")
        e = int(np.argmin(h["val_loss"]))
        ax.scatter([e + 1], [h["val_loss"][e]], color="#e8743b", zorder=5)
        late = e + 1 > 0.7 * len(ep)
        ax.annotate(f"min val loss\nepoch {e + 1}", (e + 1, h["val_loss"][e]),
                    textcoords="offset points", xytext=(-10 if late else 8, 16),
                    ha="right" if late else "left", fontsize=8)
        ax.set_title(f"{c['name']}: {c['description']}", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss")
        ax.legend()
        _style(ax)
    fig.suptitle("Training vs validation loss", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_val_comparison(configs, path):
    """Validation accuracy and loss across epochs, all configurations together."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, c in enumerate(configs):
        h = c["history"]
        ep = np.arange(1, len(h["val_acc"]) + 1)
        ax[0].plot(ep, np.array(h["val_acc"]) * 100, color=COLORS[i % 5], lw=2, label=c["name"])
        ax[1].plot(ep, h["val_loss"], color=COLORS[i % 5], lw=2, label=c["name"])
    ax[0].set_title("Validation accuracy (%)")
    ax[1].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_best_curves(config, path):
    """Loss and accuracy curves of the selected configuration."""
    h = config["history"]
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, h["train_loss"], lw=2, color="#2f6fed", label="train")
    ax[0].plot(ep, h["val_loss"], lw=2, color="#e8743b", label="validation")
    ax[0].set_title(f"{config['name']}: loss")
    ax[1].plot(ep, np.array(h["train_acc"]) * 100, lw=2, color="#2f6fed", label="train")
    ax[1].plot(ep, np.array(h["val_acc"]) * 100, lw=2, color="#e8743b", label="validation")
    ax[1].set_title(f"{config['name']}: accuracy (%)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        _style(a)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion(cm, path):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("predicted digit")
    ax.set_ylabel("true digit")
    ax.set_title("Confusion matrix (test set)")
    thr = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_errors(images, true, pred, path, n=12):
    """Grid of misclassified test digits."""
    n = min(n, len(images))
    cols = 6
    rows = max(1, (n + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k in range(n):
        ax = axes.ravel()[k]
        ax.imshow(images[k], cmap="gray")
        ax.set_title(f"true {true[k]} / pred {pred[k]}", fontsize=9, color="#c0392b")
    fig.suptitle("Examples the best model got wrong", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def overfitting_notes(configs):
    """Plain-language, data-driven notes on over/underfitting for each config."""
    notes = []
    for c in configs:
        h = c["history"]
        n = len(h["val_loss"])
        e = int(np.argmin(h["val_loss"]))
        vmin, vfin = h["val_loss"][e], h["val_loss"][-1]
        tfin = h["train_loss"][-1]
        if e < n - 1 and vfin - vmin > 0.02:
            notes.append(
                f"- **{c['name']}**: validation loss was lowest at epoch {e + 1} ({vmin:.4f}) and "
                f"rose to {vfin:.4f} by epoch {n}, while training loss kept falling to {tfin:.4f}. "
                f"This is the overfitting pattern: the model keeps fitting the training set but "
                f"stops generalising after about epoch {e + 1}.")
        else:
            notes.append(
                f"- **{c['name']}**: no clear overfitting. Validation loss ended at {vfin:.4f} "
                f"(minimum {vmin:.4f} at epoch {e + 1}) with training loss {tfin:.4f}.")
    return "\n".join(notes)


def build_readme(res, arch_text, out_path):
    cfgs = res["configs"]
    best = next(c for c in cfgs if c["name"] == res["best"])
    t = res["test"]
    rows = "\n".join(
        f"| {c['name']} | {c['description']} | {c['n_params']:,} | {c['train_loss_final']:.4f} | "
        f"{c['val_loss_final']:.4f} | {c['val_acc_final'] * 100:.2f}% |"
        for c in cfgs)
    worst = min(cfgs, key=lambda c: c["val_acc_final"])
    pc = t["per_class_acc"]
    hardest = int(np.argmin(pc))
    easiest = int(np.argmax(pc))
    md = f"""# ML-8: Deep Learning Intro (PyTorch, MNIST)

A fully connected neural network built and trained from scratch with PyTorch `torch.nn`, with a **manual training loop** (forward pass, loss, backward pass, optimizer step). Four architecture/hyperparameter variations are compared on a validation set, and only the best one is evaluated on the test set.

## Final result

**Test accuracy of the best configuration (`{best['name']}`): {t['acc'] * 100:.2f}%** (test loss {t['loss']:.4f}, {t['n_errors']} errors out of {res['n_test']:,} images).

## Requirements checklist

| Requirement | Where |
|---|---|
| Architecture with input, hidden layers (non-linear activation) and output, using `torch.nn` | `build_model()` in `train.py` |
| Full training loop: forward, loss, backward, optimizer step (manual) | `train_one_epoch()` in `train.py` |
| Real dataset with a proper train/validation split | MNIST: {res['n_train']:,} train / {res['n_val']:,} validation / {res['n_test']:,} test |
| Training and validation loss curves across epochs | `images/loss_curves_all.png` |
| At least 2 architecture/hyperparameter variations compared on validation performance | 4 configurations (below) |
| Final test accuracy | above |

## Dataset

MNIST handwritten digits (10 classes, 28x28 grayscale), loaded with `torchvision.datasets.MNIST`. The official 60,000-image training set is split with a fixed seed into {res['n_train']:,} for training and {res['n_val']:,} for validation. The official 10,000-image test set is used **once**, at the end, for the best configuration only. Pixels are scaled to [0, 1] and normalised with the MNIST mean (0.1307) and std (0.3081), then flattened to 784 values.

## Model

Best configuration (`{best['name']}`), {best['n_params']:,} trainable parameters:

```
{arch_text}
```

## How the training loop works

For every mini-batch of {res['batch_size']} images (`train_one_epoch()` in `train.py`):

1. **Forward pass:** `logits = model(xb)` pushes the batch through the layers to produce 10 raw class scores.
2. **Loss:** `loss = nn.CrossEntropyLoss()(logits, yb)` compares the scores with the true digits (softmax + negative log-likelihood).
3. **Zero gradients:** `optimizer.zero_grad()` clears gradients from the previous step. PyTorch accumulates gradients by default, so skipping this silently corrupts training.
4. **Backward pass:** `loss.backward()` uses backpropagation to compute the gradient of the loss with respect to every weight.
5. **Optimizer step:** `optimizer.step()` updates the weights using Adam.

After each epoch the model is evaluated on the validation set with gradients disabled (`torch.no_grad()`) and in `eval()` mode.

## Experiments

All runs use the same seed ({res['seed']}), Adam optimizer, batch size {res['batch_size']} and {res['epochs']} epochs. Only the listed settings change.

| Name | Settings | Parameters | Final train loss | Final val loss | Final val accuracy |
|---|---|---|---|---|---|
{rows}

The best configuration by final validation accuracy is **`{best['name']}`** ({best['val_acc_final'] * 100:.2f}%), compared with {worst['val_acc_final'] * 100:.2f}% for the weakest (`{worst['name']}`).

![Validation comparison](images/val_comparison.png)

## Loss curves and overfitting check

![Loss curves](images/loss_curves_all.png)

{overfitting_notes(cfgs)}

Curves of the selected configuration:

![Best configuration curves](images/best_model_curves.png)

## Test results (best configuration, evaluated once)

![Confusion matrix](images/confusion_matrix.png)

The easiest digit was **{easiest}** ({pc[easiest] * 100:.2f}% correct) and the hardest was **{hardest}** ({pc[hardest] * 100:.2f}% correct).

![Misclassified examples](images/misclassified_examples.png)

## Run it

Windows PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python train.py
```

macOS / Linux:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

The script downloads MNIST into `data/`, trains all configurations, writes the plots to `images/`, the numbers to `results/results.json`, saves the best weights to `results/best_model.pt`, and regenerates this README with your results. Everything is seeded, so results are reproducible on the same machine and library versions.

## Project structure

```
train.py            # data, model, manual training loop, experiments, test evaluation
report.py           # plots + README generation (numpy/matplotlib only)
requirements.txt
images/             # all plots shown above
results/            # results.json, best_model.pt
```
"""
    Path(out_path).write_text(md, encoding="utf-8")
