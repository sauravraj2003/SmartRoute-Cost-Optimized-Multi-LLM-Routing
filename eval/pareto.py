"""Compute and optionally plot cost-quality Pareto frontiers."""
from __future__ import annotations


def pareto_frontier(points: list[dict], cost_key: str = "mean_cost_usd", quality_key: str = "success_rate") -> list[dict]:
    frontier = []
    for point in sorted(points, key=lambda p: p[cost_key]):
        if not any(other[quality_key] >= point[quality_key] and other[cost_key] <= point[cost_key]
                   and (other[quality_key] > point[quality_key] or other[cost_key] < point[cost_key]) for other in points):
            frontier.append(point)
    return frontier


def plot_pareto(points: list[dict], output: str) -> None:
    import matplotlib.pyplot as plt
    frontier = pareto_frontier(points)
    fig, ax = plt.subplots()
    ax.scatter([p["mean_cost_usd"] for p in points], [p["success_rate"] for p in points], label="Policies")
    if frontier:
        frontier.sort(key=lambda p: p["mean_cost_usd"])
        ax.plot([p["mean_cost_usd"] for p in frontier], [p["success_rate"] for p in frontier], label="Pareto frontier")
    ax.set(xlabel="Mean cost / request (USD)", ylabel="Success rate", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
