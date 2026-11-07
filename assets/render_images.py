# Render README images from the recorded experiment

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets"
INK = "#172b3a"
MUTED = "#586b78"
BLUE = "#2864b4"
TEAL = "#147d78"
ORANGE = "#bd6930"
BORDER = "#d6e0e7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": BORDER,
    "savefig.facecolor": "white",
})


def box(ax, x, y, width, height, title, detail, color=BLUE, fill="#edf4fc"):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=fill, edgecolor=BORDER, linewidth=1.2,
    ))
    ax.text(x + 0.18, y + height - 0.31, title, fontsize=14,
            weight="bold", color=color, va="center")
    ax.text(x + 0.18, y + 0.3, detail, fontsize=11, color=MUTED,
            va="center", linespacing=1.5)


def arrow(ax, start, end, color=MUTED):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                               mutation_scale=16, linewidth=1.6, color=color))


def render_architecture():
    fig = plt.figure(figsize=(12, 6.5), facecolor="white")
    fig.text(0.045, 0.92, "How image features become a caption", fontsize=22, weight="bold")
    fig.text(0.045, 0.865, "Two decoder layers combine the image with the words seen so far",
             fontsize=12, color=MUTED)
    ax = fig.add_axes([0.035, 0.08, 0.93, 0.72])
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")

    box(ax, 0.1, 3.45, 3.0, 1.1, "Caption prefix", "<START>  a  dog")
    box(ax, 3.7, 3.45, 3.1, 1.1, "Word representation", "Embeddings + positions\n256 numbers per word")
    box(ax, 0.1, 1.9, 3.0, 1.1, "Image features", "VGG16 + PCA\n512 numbers per image", TEAL, "#edf7f4")
    box(ax, 3.7, 1.9, 3.1, 1.1, "Image projection", "Linear layer\n512 → 256 numbers", TEAL, "#edf7f4")
    arrow(ax, (3.12, 4), (3.67, 4), BLUE)
    arrow(ax, (3.12, 2.45), (3.67, 2.45), TEAL)

    ax.add_patch(FancyBboxPatch((7.5, 1.85), 4.1, 2.75,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor="#f7f9fb", edgecolor=BORDER, linewidth=1.2))
    ax.text(7.7, 4.27, "Transformer decoder × 2", fontsize=14, weight="bold")
    ax.text(7.7, 3.8, "Masked self-attention", fontsize=13, color=BLUE)
    ax.text(7.7, 3.46, "Reads current and earlier words", fontsize=10, color=MUTED)
    ax.text(7.7, 2.9, "Cross-attention", fontsize=13, color=TEAL)
    ax.text(7.7, 2.56, "Reads the projected image", fontsize=10, color=MUTED)
    ax.text(7.7, 2.13, "Feedforward + residuals + layer norms", fontsize=10, color=MUTED)
    arrow(ax, (6.82, 4), (7.47, 3.8), BLUE)
    arrow(ax, (6.82, 2.45), (7.47, 2.9), TEAL)

    box(ax, 7.5, 0.1, 4.1, 1.05, "Next-word scores", "Choose a word and append it to the prefix")
    arrow(ax, (9.55, 1.83), (9.55, 1.18))
    ax.text(0.1, 0.85, "Repeat until <END> or the length limit", fontsize=15, weight="bold")
    ax.text(0.1, 0.42, "Training uses the correct prefix\nGeneration uses the model’s previous words",
            fontsize=11, color=MUTED, linespacing=1.5)
    fig.savefig(OUTPUT / "architecture.png", dpi=180)
    plt.close(fig)


def render_results(metrics):
    fig, (loss_ax, accuracy_ax) = plt.subplots(
        1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.7, 1]}, facecolor="white",
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.70, wspace=0.35)
    fig.text(0.045, 0.91, "Learning on a small caption set", fontsize=22, weight="bold")
    fig.text(0.045, 0.84,
             f"{metrics['max_train']} training captions  ·  {len(metrics['loss_history'])} updates  ·  CPU  ·  seed {metrics['seed']}",
             fontsize=12, color=MUTED)

    losses = metrics["loss_history"]
    steps = range(1, len(losses) + 1)
    loss_ax.plot(steps, losses, color=BLUE, linewidth=2.1)
    loss_ax.axhline(metrics["training_target"], color=ORANGE, linestyle="--", linewidth=1.2,
                     label=f"Target {metrics['training_target']}")
    loss_ax.scatter([len(losses)], [losses[-1]], color=BLUE, s=30, zorder=4)
    loss_ax.annotate(f"Final loss {losses[-1]:.4f}", xy=(len(losses), losses[-1]),
                     xytext=(105, 1.8), color=BLUE, fontsize=12, weight="bold",
                     arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2})
    loss_ax.set(title="Training loss", xlabel="Training update", ylabel="Masked cross-entropy",
                xlim=(0, len(losses) + 5), ylim=(-0.18, 5.5))
    loss_ax.set_xticks([0, 50, 100, 150, 200])
    loss_ax.grid(axis="y", color=BORDER, linewidth=0.6, alpha=0.7)
    loss_ax.set_axisbelow(True)
    loss_ax.legend(frameon=False, loc="upper right", fontsize=10)

    accuracy = [metrics["train_eval"]["token_accuracy"], metrics["validation_eval"]["token_accuracy"]]
    bars = accuracy_ax.bar(["Training", "Validation"], accuracy, color=[BLUE, ORANGE], width=0.55)
    for bar, value in zip(bars, accuracy):
        label = f"{value:.0%}" if value == 1 else f"{value:.2%}"
        accuracy_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, label,
                         ha="center", fontsize=14, weight="bold")
    accuracy_ax.set(title="Next-word accuracy", ylim=(0, 1.2))
    accuracy_ax.yaxis.set_major_formatter(PercentFormatter(1))
    accuracy_ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    accuracy_ax.grid(axis="y", color=BORDER, linewidth=0.6, alpha=0.7)
    accuracy_ax.set_axisbelow(True)
    accuracy_ax.tick_params(axis="x", length=0)

    fig.text(0.08, 0.08, "Recorded minibatch losses from the baseline run", fontsize=10, color=MUTED)
    fig.text(0.64, 0.08, "50 captions per split · correct prefixes\nPadding excluded from accuracy",
             fontsize=10, color=MUTED, linespacing=1.5)
    fig.savefig(OUTPUT / "training-results.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    metrics = json.loads((ROOT / "reports" / "coco_run.json").read_text())
    render_architecture()
    render_results(metrics)
    print("Rendered architecture and results images")
