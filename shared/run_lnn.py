# LNN (Category C) on MNIST addition.
# LNN can't backprop into a CNN, so the CNN is trained separately on digit labels and
# LNN only does the addition reasoning: Sum_s = OR over {a+b=s} of (digit1_a AND digit2_b).
# Reported separately from A/B since it isn't trained end-to-end.
# Usage: python run_lnn.py [--eval_instances 1000]
import argparse
import csv
import os
import sys
import time
from datetime import datetime

SHARED = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SHARED)
sys.path.insert(0, SHARED)

import torch
import torchvision
from torchvision.transforms import transforms
from lnn import Model, Proposition, And, Or, Direction

from shared_cnn import SharedCNN
import eval_utils as E

SEED = 42
DEVICE = "cpu"
NEURASP_DATA = os.path.join(PROJECT, "repos", "NeurASP", "examples", "data")


def train_cnn_on_digits(epochs=1):
    """Train the shared CNN on MNIST digit labels (decoupled perception)."""
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((E.MNIST_MEAN,), (E.MNIST_STD,))])
    train = torchvision.datasets.MNIST(NEURASP_DATA, train=True, download=True, transform=tfm)
    loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    net = SharedCNN(with_softmax=False).to(DEVICE)   # logits for cross-entropy
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    net.train()
    t0 = time.time()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            lossf(net(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
    return net, time.time() - t0


def build_lnn():
    """LNN model: Sum_s = OR_{a+b=s} (digit1_a AND digit2_b)."""
    m = Model()
    d1 = [Proposition("d1_{}".format(a)) for a in range(10)]
    d2 = [Proposition("d2_{}".format(b)) for b in range(10)]
    sums = []
    for s in range(19):
        terms = [And(d1[a], d2[b]) for a in range(10) for b in range(10) if a + b == s]
        S = Or(*terms) if len(terms) > 1 else terms[0]
        m.add_knowledge(S)
        sums.append(S)
    return m, d1, d2, sums


def lnn_predict_sum(m, d1, d2, sums, a_pred, b_pred):
    """Feed the CNN's predicted digits as one-hot facts; LNN infers the sum."""
    m.flush()
    data = {d1[a]: (1.0 if a == a_pred else 0.0,) * 2 for a in range(10)}
    data.update({d2[b]: (1.0 if b == b_pred else 0.0,) * 2 for b in range(10)})
    m.add_data(data)
    m.infer(direction=Direction.UPWARD)
    truths = [sums[s].get_data()[0].item() for s in range(19)]
    return int(max(range(19), key=lambda s: truths[s]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_instances", type=int, default=1000)
    args = ap.parse_args()

    E.set_seed(SEED)
    print("[lnn] training decoupled CNN perception on MNIST digits...", flush=True)
    net, cnn_time = train_cnn_on_digits(epochs=1)

    test_images, test_labels = E.load_mnist_test(DEVICE)
    preds = E.predict_digits(net, test_images, DEVICE)
    digit_acc = E.digit_accuracy(preds, test_labels)
    preds = preds.tolist()
    labels = test_labels.tolist()

    m, d1, d2, sums = build_lnn()
    n_inst = min(args.eval_instances, len(preds) // 2)
    correct = 0
    t0 = time.time()
    for i in range(n_inst):
        a_pred, b_pred = preds[2 * i], preds[2 * i + 1]
        true_sum = labels[2 * i] + labels[2 * i + 1]
        if lnn_predict_sum(m, d1, d2, sums, a_pred, b_pred) == true_sum:
            correct += 1
    lnn_time = time.time() - t0
    sum_acc = 100.0 * correct / n_inst
    ms_per_inst = 1000.0 * lnn_time / n_inst

    out = os.path.join(PROJECT, "results", "lnn_result.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["framework", "category", "n_digits", "sum_acc", "digit_acc",
                    "cnn_train_time_s", "lnn_inference_ms_per_instance", "eval_instances",
                    "mode", "seed", "timestamp"])
        w.writerow(["LNN", "C (structure/architecture)", 1, round(sum_acc, 3), round(digit_acc, 3),
                    round(cnn_time, 2), round(ms_per_inst, 2), n_inst,
                    "reasoning-only: CNN trained separately (LNN cannot backprop to CNN)",
                    SEED, datetime.now().isoformat(timespec="seconds")])

    print("[lnn] CNN digit acc: {:.2f}%   (trained {:.1f}s)".format(digit_acc, cnn_time), flush=True)
    print("[lnn] sum acc over {} instances: {:.2f}%".format(n_inst, sum_acc), flush=True)
    print("[lnn] LNN inference: {:.1f} ms/instance".format(ms_per_inst), flush=True)
    print("[lnn] wrote", out, flush=True)


if __name__ == "__main__":
    main()
