import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_map(env, agent, output_path):
    world = np.zeros((env.height, env.width))

    for x, y in env.obstacles:
        world[y, x] = 1

    for x, y in env.visited:
        if world[y, x] == 0:
            world[y, x] = 2

    for x, y in env.scanned:
        if world[y, x] == 2:
            world[y, x] = 3

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(
        world,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=3
    )

    ax.imshow(
        agent.belief.T,
        origin="lower",
        cmap="YlOrRd",
        alpha=0.48
    )

    ax.scatter(
        [env.start[0]], [env.start[1]],
        marker="s", s=140, label="Base"
    )

    ax.scatter(
        [agent.position[0]], [agent.position[1]],
        marker="o", s=110, label="WALL-E"
    )

    estimate = agent.best_estimate()["location"]

    ax.scatter(
        [estimate[0]], [estimate[1]],
        marker="*", s=230, label="Best EVE estimate"
    )

    if env.eve_found:
        ax.scatter(
            [env.position[0]], [env.position[1]],
            marker="X", s=160, label="EVE located"
        )

    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("WALL-E and EVE's Signal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)