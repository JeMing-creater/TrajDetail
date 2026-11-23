"""Plot the five-cohort benchmark from the released aggregate CSV."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np


def main():
    output = Path(__file__).resolve().parent
    with (output.parent / "nrmse_3fold.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    cohorts = ["DLBS", "MU-Glioma-Post", "ACDC", "LUMIERE", "ISLES’24"]
    labels = {
        "Last image (LCI)": "Last image (LCI)",
        "BrLP (LAS + validation-selected decoder)": "BrLP (revised)",
        "TrajDetail (ours)": "TrajDetail",
    }
    names = [labels.get(row["method"], row["method"]) for row in rows]
    means = np.array([[float(row[f"{cohort} mean"]) for cohort in cohorts] for row in rows])
    relative = 100 * means / means[names.index("Last image (LCI)")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.2, 5.7), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("improvement", ["#0072B2", "#FAFAFA", "#E69F00"])
    norm = TwoSlopeNorm(vmin=relative.min(), vcenter=100, vmax=relative.max())
    image = ax.imshow(relative, cmap=cmap, norm=norm, aspect="auto")
    for row, values in enumerate(relative):
        for column, value in enumerate(values):
            color = "white" if norm(value) < 0.24 else "#172834"
            ax.text(column, row, f"{value:.0f}%", ha="center", va="center", color=color)

    ours = names.index("TrajDetail")
    ax.add_patch(Rectangle((-0.49, ours - 0.48), len(cohorts) - 0.02, 0.96,
                           fill=False, edgecolor="#172834", linewidth=1.5))
    ax.set_xticks(range(len(cohorts)), ["DLBS", "MU-Glioma\nPost", "ACDC", "LUMIERE", "ISLES’24"])
    ax.set_yticks(range(len(names)), names)
    ax.get_yticklabels()[ours].set_fontweight("bold")
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Three-fold NRMSE relative to copying the last image",
                 loc="left", weight="bold", pad=14)
    bar = fig.colorbar(image, ax=ax, pad=0.03, shrink=0.85, ticks=[80, 100, 150, 200, 300])
    bar.set_label("Relative NRMSE (%) · lower is better")
    bar.ax.axhline(100, color="#172834", linewidth=0.8)

    fig.savefig(output / "benchmark_relative_nrmse.png", dpi=300, facecolor="white")
    fig.savefig(output / "benchmark_relative_nrmse.pdf", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
