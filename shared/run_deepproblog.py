# DeepProbLog (Category B) on MNIST addition.
# Trains the shared CNN with the exact (SDD) engine; a hook evaluates accuracy every
# EVAL_EVERY pairs and enforces the budget and a time cap.
# Usage: python run_deepproblog.py --method {exact,geometric_mean} --n {1,2,3}
import argparse
import os
import sys
import time
import warnings
from datetime import datetime

warnings.simplefilter("ignore")
SHARED = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(SHARED)
sys.path.insert(0, SHARED)   # shared_cnn, eval_utils

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ExactEngine, ApproximateEngine
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import TrainObject

from shared_cnn import SharedCNN
import eval_utils as E

BUDGET_PAIRS = int(os.environ.get("BUDGET_PAIRS", 15000))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", 1500))      # pairs between evaluations
TIME_CAP_S = float(os.environ.get("TIME_CAP_S", 1200))    # wall-clock cap (excludes eval time)
SEED = 42
ACC90 = 90.0
BATCH = 2
DEVICE = "cpu"
MODEL_PL = os.path.join(PROJECT, "repos", "deepproblog", "src", "deepproblog",
                        "examples", "MNIST", "models", "addition.pl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["exact", "geometric_mean"])
    ap.add_argument("--n", type=int, required=True, choices=[1, 2, 3])
    args = ap.parse_args()
    method, n = args.method, args.n
    short = "exact" if method == "exact" else "approx"
    setup = "dpl_{}_n{}".format(short, n)
    dataset_size = {1: 30000, 2: 15000, 3: 10000}[n]

    E.set_seed(SEED)
    E.reset_setup(setup, n)   # overwrite any previous output for this setup
    test_images, test_labels = E.load_mnist_test(DEVICE)

    train_set = addition(n, "train")
    network = SharedCNN().to(DEVICE)
    net = Network(network, "mnist_net", batching=True)
    net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    model = Model(MODEL_PL, [net])
    if method == "exact":
        model.set_engine(ExactEngine(model), cache=True)
    else:
        model.set_engine(ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False))
    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    loader = DataLoader(train_set, BATCH, False)

    state = {"eval_time": 0.0, "iters_to_90": "", "time_to_90": "",
             "status": "completed", "pairs_seen": 0, "train_time": 0.0,
             "sum_acc": 0.0, "digit_acc": 0.0}
    t_start = time.time()

    sa, da = E.evaluate(network, test_images, test_labels, n, DEVICE)
    E.append_curve_point(setup, n, dict(pairs_seen=0, train_time_s=0.0,
                                        sum_acc=round(sa, 3), digit_acc=round(da, 3), loss=""))
    print("[{}] init sum={:.2f} digit={:.2f}".format(setup, sa, da), flush=True)

    def hook(train_obj):
        pairs = train_obj.i * BATCH
        e0 = time.time()
        sa, da = E.evaluate(network, test_images, test_labels, n, DEVICE)
        state["eval_time"] += time.time() - e0
        train_time = (time.time() - t_start) - state["eval_time"]
        loss = train_obj.accumulated_loss
        E.append_curve_point(setup, n, dict(pairs_seen=pairs, train_time_s=round(train_time, 2),
                                            sum_acc=round(sa, 3), digit_acc=round(da, 3),
                                            loss=round(float(loss), 4) if loss else ""))
        print("[{}] pairs={} time={:.1f}s sum={:.2f} digit={:.2f}".format(
            setup, pairs, train_time, sa, da), flush=True)
        state.update(pairs_seen=pairs, train_time=train_time, sum_acc=sa, digit_acc=da)
        if state["iters_to_90"] == "" and sa >= ACC90:
            state["iters_to_90"] = pairs
            state["time_to_90"] = round(train_time, 2)
        if pairs >= BUDGET_PAIRS:
            train_obj.interrupt = True
        elif train_time >= TIME_CAP_S:
            state["status"] = "time_capped"
            train_obj.interrupt = True

    to = TrainObject(model)
    to.hooks.append((EVAL_EVERY // BATCH, hook))

    try:
        to.train(loader, 1, log_iter=100, initial_test=False)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
    finally:
        sa, da = E.evaluate(network, test_images, test_labels, n, DEVICE)
        state.update(sum_acc=sa, digit_acc=da)
        if state["pairs_seen"] == 0:
            state["pairs_seen"] = to.i * BATCH
            state["train_time"] = (time.time() - t_start) - state["eval_time"]
        E.append_summary(dict(
            setup=setup, framework="DeepProbLog",
            method=("exact" if method == "exact" else "approximate(geom_mean)"),
            n_digits=n, train_pairs_budget=BUDGET_PAIRS, batch_size=BATCH,
            epochs_equiv=round(state["pairs_seen"] / dataset_size, 3),
            final_sum_acc=round(sa, 3), final_digit_acc=round(da, 3),
            total_train_time_s=round(state["train_time"], 2),
            time_per_pass_s=round(state["train_time"] / max(state["pairs_seen"] / dataset_size, 1e-9), 2),
            iters_to_90_pairs=state["iters_to_90"], time_to_90_s=state["time_to_90"],
            status=state["status"], seed=SEED,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        ))
        print("[{}] DONE status={} sum={:.2f} digit={:.2f} time={:.1f}s".format(
            setup, state["status"], sa, da, state["train_time"]), flush=True)


if __name__ == "__main__":
    main()
