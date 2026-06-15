# Build the summary table and the plots from the CSV files in results/.
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(PROJECT, "results")
PLOTS = os.path.join(PROJECT, "plots")
os.makedirs(PLOTS, exist_ok=True)

FAMILY = {
    "neurasp": ("NeurASP (Cat A)", "#1f77b4"),
    "dpl_exact": ("DeepProbLog (Cat B)", "#2ca02c"),
}
ORDER = ["neurasp", "dpl_exact"]
NS = [1, 2, 3]


def family_of(s):
    for k in ORDER:
        if s.startswith(k):
            return k
    return s


def fnum(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def load_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def load_curve(setup, n):
    with open(os.path.join(RES, "curve_{}_n{}.csv".format(setup, n))) as f:
        return list(csv.DictReader(f))


def row_for(rows, fam, n):
    return next((x for x in rows if family_of(x["setup"]) == fam and x["n_digits"] == str(n)), None)


# summary table
def summary_table(rows):
    cols = ["Category / framework", "Digits", "Sum acc (%)", "Digit acc (%)",
            "Total time (s)", "Iters to 90% sum (pairs)"]
    table = []
    for fam in ORDER:
        for n in NS:
            r = row_for(rows, fam, n)
            if not r:
                continue
            table.append([FAMILY[fam][0], n, r["final_sum_acc"], r["final_digit_acc"],
                          r["total_train_time_s"], r["iters_to_90_pairs"] or "n/a"])

    # add LNN (Cat C) row if present; it is reasoning-only, not comparable to A/B
    lnn_path = os.path.join(RES, "lnn_result.csv")
    lnn_row_idx = None
    if os.path.isfile(lnn_path):
        L = load_csv("lnn_result.csv")[0]
        lnn_row_idx = len(table)
        table.append(["LNN (Cat C) *", 1, L["sum_acc"], L["digit_acc"], "separate", "n/a"])

    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(str(x) for x in t) + " |" for t in table]
    with open(os.path.join(RES, "summary_table.md"), "w") as f:
        f.write("# MNIST addition: Categories A, B, C summary\n\n"
                "Shared CNN, seed 42, all runs local on Apple M2 (CPU). NeurASP uses a column-wise ASP\n"
                "encoding; DeepProbLog uses its standard encoding with exact inference. Sum accuracy\n"
                "drops as digits grow because every digit must be correct (0.975^6 is about 0.86),\n"
                "while digit accuracy stays around 97%.\n\n"
                + "\n".join(md) + "\n\n"
                "DeepProbLog-exact did not run at 3 digits: its inference circuit failed to compile\n"
                "within 21 minutes. NeurASP completes 3 digits in about 5 minutes.\n\n"
                "* LNN (Category C) is reasoning-only: it cannot backprop into a CNN, so the CNN was\n"
                "trained separately on digit labels and LNN infers the sum from its predictions. This\n"
                "is not directly comparable to the end-to-end A/B runs (CNN ~6.8 s, LNN ~63 ms/instance).\n")

    fig, ax = plt.subplots(figsize=(12, 3.1))
    ax.axis("off")
    tbl = ax.table(cellText=[[str(x) for x in r] for r in table], colLabels=cols,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(col=list(range(len(cols))))
    tbl.scale(1, 1.5)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#40466e")
        tbl[0, j].set_text_props(color="w", weight="bold")
    if lnn_row_idx is not None:                       # tint the Category-C row
        for j in range(len(cols)):
            tbl[lnn_row_idx + 1, j].set_facecolor("#fdf0d5")
    ax.set_title("MNIST addition summary: Categories A, B, C  (shared CNN, seed 42, M2 CPU)",
                 pad=10, weight="bold")
    fig.text(0.5, 0.05, "DeepProbLog-exact did not run at 3 digits: inference circuit failed to compile in 21 min",
             ha="center", fontsize=8.5, style="italic", color="#900")
    fig.text(0.5, 0.005, "* LNN (Cat C) is reasoning-only: perception trained separately (LNN can't backprop to a CNN); "
             "not directly comparable to the A/B runs.  CNN ~6.8 s, LNN ~63 ms/instance.",
             ha="center", fontsize=7.5, style="italic", color="#555")
    fig.savefig(os.path.join(RES, "summary_table.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# accuracy by digit count
def plot_accuracy(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.38
    panels = [("final_sum_acc", "Sum accuracy (%)", "Sum accuracy"),
              ("final_digit_acc", "Digit accuracy (%)", "Digit (recognition) accuracy")]
    for ax, (key, ylab, title) in zip(axes, panels):
        for fi, fam in enumerate(ORDER):
            labelled = False
            for ni, n in enumerate(NS):
                r = row_for(rows, fam, n)
                xpos = ni + (fi - 0.5) * width
                if r:
                    v = fnum(r[key])
                    ax.bar(xpos, v, width, color=FAMILY[fam][1], edgecolor="black",
                           label=None if labelled else FAMILY[fam][0])
                    labelled = True
                    ax.text(xpos, v + 0.8, "{:.1f}".format(v), ha="center", fontsize=8)
                else:
                    ax.text(xpos, 50, "✗\ndid\nnot\nrun", ha="center", va="center",
                            fontsize=8.5, color=FAMILY[fam][1], weight="bold")
        ax.axhline(90, color="gray", ls=":", lw=1)
        ax.set_xticks(range(len(NS)))
        ax.set_xticklabels(["{} digit{}".format(n, "s" if n > 1 else "") for n in NS])
        ax.set_xlim(-0.6, len(NS) - 1 + 0.6)   # right margin so the N=3 "did not run" label fits inside
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 108)
        ax.set_title(title)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Accuracy by digit count  (DeepProbLog-exact cannot run at 3 digits)", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "accuracy_by_setup.png"), dpi=150)
    plt.close(fig)


# compute cost vs digit count
def plot_compute(cost):
    fig, ax = plt.subplots(figsize=(9, 6))
    WALL = 1.3e6  # 1300 s in ms (timeout DPL-exact n=3 hit without compiling)
    for fam in ORDER:
        pts = sorted([c for c in cost if family_of(c["setup"]) == fam], key=lambda c: int(c["n_digits"]))
        xs, ys = [], []
        for c in pts:
            ms = fnum(c["ms_per_step"])
            if ms is not None:
                xs.append(int(c["n_digits"]))
                ys.append(ms)
        ax.plot(xs, ys, "-o", color=FAMILY[fam][1], label=FAMILY[fam][0])
        for x, y in zip(xs, ys):
            ax.annotate("{:.1f} ms".format(y), (x, y), textcoords="offset points",
                        xytext=(7, 5), fontsize=8, color=FAMILY[fam][1])
        n3 = next((c for c in pts if c["n_digits"] == "3"), None)
        if n3 and fnum(n3["ms_per_step"]) is None:
            ax.plot([xs[-1], 3], [ys[-1], WALL], ":", color=FAMILY[fam][1])
            ax.plot([3], [WALL], "x", ms=16, mew=3, color=FAMILY[fam][1])
            ax.annotate("✗ circuit did not\ncompile (>21 min)", (3, WALL),
                        textcoords="offset points", xytext=(-4, -46), fontsize=9,
                        color=FAMILY[fam][1], ha="center", weight="bold")
    ax.axhspan(1e5, 1e7, color="red", alpha=0.05)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Digits per number (N)")
    ax.set_ylabel("Time per training step (ms, log scale)")
    ax.set_title("Computational cost vs problem size\n"
                 "NeurASP (Cat A) stays tractable to 3 digits; DeepProbLog-exact (Cat B) cannot run")
    ax.legend(loc="center left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "compute_cost_vs_digits.png"), dpi=150)
    plt.close(fig)


# learning curves
def plot_curves(rows):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, n in zip(axes, NS):
        for fam in ORDER:
            r = row_for(rows, fam, n)
            if not r:
                continue
            curve = load_curve(r["setup"], n)
            xs = [fnum(c["pairs_seen"]) for c in curve]
            ys = [fnum(c["sum_acc"]) for c in curve]
            ax.plot(xs, ys, "-o", ms=4, color=FAMILY[fam][1], label=FAMILY[fam][0])
        ax.axhline(90, color="gray", ls=":", lw=1)
        ax.set_title("{} digit{}".format(n, "s" if n > 1 else ""))
        ax.set_xlabel("Training pairs seen")
        ax.grid(alpha=0.3)
        if n == 3:
            ax.text(0.5, 0.45, "DeepProbLog-exact:\ncircuit did not compile\n(no training possible)",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9.5,
                    color=FAMILY["dpl_exact"][1],
                    bbox=dict(boxstyle="round", fc="white", ec=FAMILY["dpl_exact"][1]))
    axes[0].set_ylabel("Sum accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Learning curves: sum accuracy vs training pairs", weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, "learning_curves.png"), dpi=150)
    plt.close(fig)


def plot_matrix(rows):
    """Matrix of the three categories x digit counts, showing each cell's status
    (ran / ran with caveat / failed)."""
    lnn = load_csv("lnn_result.csv")[0] if os.path.isfile(os.path.join(RES, "lnn_result.csv")) else None
    GREEN, RED, AMBER, GREY = "#c8e6c9", "#ffcdd2", "#fff3c4", "#eeeeee"

    def ab_cell(fam, n):
        r = row_for(rows, fam, n)
        if r:
            return ("✓  sum {:.1f}%\n{:.0f}s".format(fnum(r["final_sum_acc"]), fnum(r["total_train_time_s"])), GREEN)
        return ("✗  inference circuit\nnever compiled\n(>21 min)", RED)

    text = [[None] * 3 for _ in range(3)]
    color = [[None] * 3 for _ in range(3)]
    for j, n in enumerate(NS):
        text[0][j], color[0][j] = ab_cell("neurasp", n)
        text[1][j], color[1][j] = ab_cell("dpl_exact", n)
    # Category C (LNN): only the 1-digit reasoning-only demonstration
    if lnn:
        text[2][0], color[2][0] = ("◐  sum {:.1f}%\nreasoning-only".format(fnum(lnn["sum_acc"])), AMBER)
    else:
        text[2][0], color[2][0] = ("◐ reasoning-only", AMBER)
    text[2][1], color[2][1] = ("reasoning-only mode\n(cannot train\nend-to-end)", GREY)
    text[2][2], color[2][2] = ("reasoning-only mode\n(cannot train\nend-to-end)", GREY)

    rowlab = ["NeurASP\nCat A (constraint)", "DeepProbLog\nCat B (proof)", "LNN\nCat C (structure)"]
    collab = ["1 digit", "2 digits", "3 digits"]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    tbl = ax.table(cellText=text, rowLabels=rowlab, colLabels=collab,
                   cellLoc="center", loc="center", cellColours=color)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 3.0)
    for j in range(3):
        tbl[0, j].set_text_props(weight="bold")
    for i in range(3):
        tbl[i + 1, -1].set_text_props(weight="bold")
    ax.set_title("Neuro-symbolic categories on MNIST addition: where each architecture works and where it breaks",
                 weight="bold", pad=16)
    fig.text(0.5, 0.055,
             "✓ trained end-to-end from sum-only labels    ✗ could not run    "
             "◐ ran in reasoning-only mode (perception trained separately)",
             ha="center", fontsize=8.5)
    fig.text(0.5, 0.01,
             "A/B values: final sum accuracy + total training time.  Category C (LNN) cannot backprop to a CNN, "
             "so it runs decoupled and is not directly comparable to A/B.",
             ha="center", fontsize=7.5, style="italic", color="#555")
    fig.savefig(os.path.join(PLOTS, "capability_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    rows = load_csv("results.csv")
    cost = load_csv("compute_cost.csv")
    summary_table(rows)
    plot_accuracy(rows)
    plot_compute(cost)
    plot_curves(rows)
    plot_matrix(rows)

    print("=== VERIFICATION: results.csv finals vs learning-curve endpoints ===")
    ok = True
    for r in rows:
        n = int(r["n_digits"])
        last = fnum(load_curve(r["setup"], n)[-1]["sum_acc"])
        summ = fnum(r["final_sum_acc"])
        m = abs(last - summ) < 0.5
        ok = ok and m
        print("  {:14s} n={}  summary={:.2f}  curve_end={:.2f}  {}".format(
            r["setup"], n, summ, last, "OK" if m else "*** MISMATCH ***"))
    print("ALL MATCH" if ok else "SOME MISMATCH")
    print("\n=== FILES ===")
    for p in ["results/summary_table.png", "results/summary_table.md",
              "plots/accuracy_by_setup.png", "plots/compute_cost_vs_digits.png",
              "plots/learning_curves.png"]:
        print("  {}  ({} bytes)".format(p, os.path.getsize(os.path.join(PROJECT, p))))


if __name__ == "__main__":
    main()
