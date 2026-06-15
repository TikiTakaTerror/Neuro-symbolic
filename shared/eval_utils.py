# Shared evaluation and CSV logging.
# Sum and digit accuracy are computed the same way for every framework, straight
# from the CNN's predictions, so the numbers are comparable.
import csv
import os
import random
import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT, "shared", "mnist_data")
RESULTS_DIR = os.path.join(PROJECT, "results")

SUMMARY_FIELDS = [
    "setup", "framework", "method", "n_digits", "train_pairs_budget",
    "batch_size", "epochs_equiv", "final_sum_acc", "final_digit_acc",
    "total_train_time_s", "time_per_pass_s", "iters_to_90_pairs",
    "time_to_90_s", "status", "seed", "timestamp",
]
CURVE_FIELDS = ["setup", "n_digits", "pairs_seen", "train_time_s",
                "sum_acc", "digit_acc", "loss"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def load_mnist_test(device="cpu"):
    """Return (images, labels) for the full MNIST test split, normalised as in training."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))]
    )
    ds = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds))
    images, labels = next(iter(loader))
    return images.to(device), labels


@torch.no_grad()
def predict_digits(net, images, device="cpu", bs=2000):
    """Argmax digit prediction over a batch of images."""
    was_training = net.training
    net.eval()
    preds = []
    for i in range(0, len(images), bs):
        out = net(images[i:i + bs].to(device))
        preds.append(out.argmax(dim=-1).detach().cpu())
    if was_training:
        net.train()
    return torch.cat(preds)


def digit_accuracy(preds, labels):
    return 100.0 * (preds == labels.cpu()).float().mean().item()


def sum_accuracy(preds, labels, n_digits):
    """Form addition instances from consecutive test images (2*n_digits images each:
    the first n form number 1, the next n form number 2) and compare the predicted
    sum (from CNN argmax) to the true sum (from the real labels)."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    per_inst = 2 * n_digits
    m = (len(preds) // per_inst) * per_inst
    p = preds[:m].reshape(-1, per_inst)
    t = labels[:m].reshape(-1, per_inst)

    def value(block):
        n = n_digits
        weights = np.array([10 ** (n - 1 - j) for j in range(n)])
        num1 = (block[:, :n] * weights).sum(axis=1)
        num2 = (block[:, n:] * weights).sum(axis=1)
        return num1 + num2

    return 100.0 * (value(p) == value(t)).mean()


def evaluate(net, test_images, test_labels, n_digits, device="cpu"):
    preds = predict_digits(net, test_images, device)
    return sum_accuracy(preds, test_labels, n_digits), digit_accuracy(preds, test_labels)


def _append_csv(path, fields, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
        f.flush()


def reset_setup(setup, n_digits):
    """Remove this setup's old output (curve file + results.csv row) so a re-run
    overwrites instead of appending."""
    cp = curve_path(setup, n_digits)
    if os.path.isfile(cp):
        os.remove(cp)
    rp = os.path.join(RESULTS_DIR, "results.csv")
    if os.path.isfile(rp):
        with open(rp) as f:
            lines = f.readlines()
        if lines:
            header, body = lines[0], lines[1:]
            body = [ln for ln in body if not ln.startswith(setup + ",")]
            with open(rp, "w") as f:
                f.write(header)
                f.writelines(body)


def append_summary(row):
    _append_csv(os.path.join(RESULTS_DIR, "results.csv"), SUMMARY_FIELDS, row)


def curve_path(setup, n_digits):
    return os.path.join(RESULTS_DIR, "curve_{}_n{}.csv".format(setup, n_digits))


def append_curve_point(setup, n_digits, point):
    point = dict(point, setup=setup, n_digits=n_digits)
    _append_csv(curve_path(setup, n_digits), CURVE_FIELDS, point)
