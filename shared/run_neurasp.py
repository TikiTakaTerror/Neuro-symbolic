# NeurASP (Category A) on MNIST addition.
# Trains the shared CNN under the ASP sum constraint, evaluating every EVAL_EVERY
# pairs to build a learning curve.
# Usage: python run_neurasp.py --n {1,2,3}
import argparse
import os
import sys
import time
from datetime import datetime

SHARED = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SHARED)
sys.path.insert(0, os.path.join(PROJECT, "repos", "NeurASP"))   # neurasp.py, mvpp.py
sys.path.insert(0, SHARED)                                       # shared_cnn, eval_utils

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from neurasp import NeurASP
from shared_cnn import SharedCNN
import eval_utils as E

BUDGET_PAIRS = int(os.environ.get("BUDGET_PAIRS", 15000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 1500))
TIME_CAP_S = float(os.environ.get("TIME_CAP_S", 1200))
SEED = 42
ACC90 = 90.0
DEVICE = "cpu"

NEURASP_DATA = os.path.join(PROJECT, "repos", "NeurASP", "examples", "data")


def gen_program(n):
    """Column-wise (carry) ASP encoding of n-digit addition.

    Adding one column at a time with a carry keeps clingo's grounding small, which
    is what lets the larger digit counts run. Number 1 = images i1..in, number 2 =
    i(n+1)..i(2n); column j counts from the units (j=0).
    """
    imgs = "; ".join("i{}".format(k) for k in range(1, 2 * n + 1))
    L = [
        "img({}).".format(imgs),
        "nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).",
        "carry(0,0).",
    ]
    for j in range(n):
        a = "i{}".format(n - j)        # number-1 digit at this column
        b = "i{}".format(2 * n - j)    # number-2 digit at this column
        L.append("colsum({j},S) :- digit(0,{a},A), digit(0,{b},B), carry({j},Cin), S=A+B+Cin.".format(j=j, a=a, b=b))
        L.append("s({j},D) :- colsum({j},S), D = S \\ 10.".format(j=j))
        L.append("carry({j1},C) :- colsum({j},S), C = S / 10.".format(j1=j + 1, j=j))
    terms = " + ".join("D{j}*{p}".format(j=j, p=10 ** j) for j in range(n))
    sterms = ", ".join("s({j},D{j})".format(j=j) for j in range(n))
    L.append("addition(M) :- {s}, carry({n},C{n}), M = {terms} + C{n}*{p}.".format(
        s=sterms, n=n, terms=terms, p=10 ** n))
    return "\n".join(L) + "\n"


class MNISTAdd(Dataset):
    def __init__(self, mnist, examples, n):
        self.mnist = mnist
        self.n = n
        self.rows = []
        with open(examples) as f:
            for line in f:
                self.rows.append([int(x) for x in line.strip().split(" ")])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        imgs, label = r[:-1], r[-1]
        return [self.mnist[j][0] for j in imgs], label


def build_dataset(n):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((E.MNIST_MEAN,), (E.MNIST_STD,))]
    )
    mnist = torchvision.datasets.MNIST(root=NEURASP_DATA, train=True, download=True, transform=transform)
    fname = {1: "mnistAdd_train.txt", 2: "mnistAdd2_train.txt", 3: "mnistAdd3_train.txt"}[n]
    examples = os.path.join(NEURASP_DATA, fname)
    ds = MNISTAdd(mnist, examples, n)
    keys = ["i{}".format(j + 1) for j in range(2 * n)]
    data = []
    for imgs, label in ds:
        d = {k: imgs[j].unsqueeze(0) for j, k in enumerate(keys)}
        obs = ":- not addition({}).".format(label)
        data.append((d, obs))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, choices=[1, 2, 3])
    args = ap.parse_args()
    n = args.n
    setup = "neurasp_n{}".format(n)

    E.set_seed(SEED)
    E.reset_setup(setup, n)   # overwrite any previous output for this setup
    test_images, test_labels = E.load_mnist_test(DEVICE)
    dataset_size = {1: 30000, 2: 15000, 3: 10000}[n]

    print("[neurasp n={}] building dataset...".format(n), flush=True)
    data = build_dataset(n)[:BUDGET_PAIRS]

    net = SharedCNN().to(DEVICE)
    optimizers = {"digit": torch.optim.Adam(net.parameters(), lr=1e-3)}
    obj = NeurASP(gen_program(n), {"digit": net}, optimizers)

    # initial (untrained) point
    sum_acc, digit_acc = E.evaluate(net, test_images, test_labels, n, DEVICE)
    E.append_curve_point(setup, n, dict(pairs_seen=0, train_time_s=0.0,
                                        sum_acc=round(sum_acc, 3), digit_acc=round(digit_acc, 3), loss=""))
    print("[neurasp n={}] init sum={:.2f} digit={:.2f}".format(n, sum_acc, digit_acc), flush=True)

    train_time = 0.0
    pairs_seen = 0
    iters_to_90 = ""
    time_to_90 = ""
    status = "completed"

    for start in range(0, len(data), EVAL_EVERY):
        chunk = data[start:start + EVAL_EVERY]
        t0 = time.time()
        obj.learn(chunk, epoch=1, storeSM=False, bar=False, task="mnistAdd_n{}".format(n))
        train_time += time.time() - t0
        pairs_seen += len(chunk)

        sum_acc, digit_acc = E.evaluate(net, test_images, test_labels, n, DEVICE)
        E.append_curve_point(setup, n, dict(pairs_seen=pairs_seen, train_time_s=round(train_time, 2),
                                            sum_acc=round(sum_acc, 3), digit_acc=round(digit_acc, 3), loss=""))
        print("[neurasp n={}] pairs={} time={:.1f}s sum={:.2f} digit={:.2f}".format(
            n, pairs_seen, train_time, sum_acc, digit_acc), flush=True)

        if iters_to_90 == "" and sum_acc >= ACC90:
            iters_to_90 = pairs_seen
            time_to_90 = round(train_time, 2)
        if train_time >= TIME_CAP_S:
            status = "time_capped"
            break

    sum_acc, digit_acc = E.evaluate(net, test_images, test_labels, n, DEVICE)
    E.append_summary(dict(
        setup=setup, framework="NeurASP", method="constraint", n_digits=n,
        train_pairs_budget=BUDGET_PAIRS, batch_size=1,
        epochs_equiv=round(pairs_seen / dataset_size, 3),
        final_sum_acc=round(sum_acc, 3), final_digit_acc=round(digit_acc, 3),
        total_train_time_s=round(train_time, 2),
        time_per_pass_s=round(train_time / max(pairs_seen / dataset_size, 1e-9), 2),
        iters_to_90_pairs=iters_to_90, time_to_90_s=time_to_90,
        status=status, seed=SEED, timestamp=datetime.now().isoformat(timespec="seconds"),
    ))
    print("[neurasp n={}] DONE status={} sum={:.2f} digit={:.2f} time={:.1f}s".format(
        n, status, sum_acc, digit_acc, train_time), flush=True)


if __name__ == "__main__":
    main()
