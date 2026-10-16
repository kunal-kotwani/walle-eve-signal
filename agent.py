import math
from collections import deque
import numpy as np


class WallEAgent:
    def __init__(self, config):
        self.width = config["map"]["width"]
        self.height = config["map"]["height"]
        self.start = tuple(config["initial_position"])
        self.position = self.start

        resources = config["resources"]
        self.battery = resources["battery"]
        self.max_battery = resources["max_battery"]
        self.time_left = resources["time_budget"]
        self.move_cost = resources["move_cost"]
        self.scan_cost = resources["scan_cost"]

        self.confidence_threshold = config["agent"][
            "confidence_threshold"
        ]

        self.visited = {self.start}
        self.scanned = set()
        self.known_obstacles = set()
        self.clue_evidence = {}
        self.trace = []

        self.belief = np.ones(
            (self.width, self.height)
        )
        self.belief[self.start] = 0
        self._normalise()

    def _normalise(self):
        for cell in self.known_obstacles:
            self.belief[cell] = 0

        self.belief = np.maximum(self.belief, 1e-12)

        for cell in self.known_obstacles:
            self.belief[cell] = 0

        total = self.belief.sum()
        if total > 0:
            self.belief /= total

    def update_belief(self, observation):
        position = tuple(
            observation.get(
                "position_after",
                observation.get("position", self.position)
            )
        )
        self.position = position

        if "signal_strength" in observation:
            observed = float(
                observation["signal_strength"]
            )

            for x in range(self.width):
                for y in range(self.height):
                    if (x, y) in self.known_obstacles:
                        continue

                    distance = (
                        abs(position[0] - x)
                        + abs(position[1] - y)
                    )

                    expected = max(
                        0, 100 - 12 * distance
                    )

                    likelihood = math.exp(
                        -((observed - expected) ** 2)
                        / (2 * 12 ** 2)
                    )

                    self.belief[x, y] *= max(
                        likelihood, 1e-8
                    )

        clue = observation.get("clue", "none")

        if clue != "none":
            weight = {
                "EVE_signal_fragment": 3.0,
                "power_residue": 1.8,
                "metal_trace": 1.25
            }.get(clue, 1.0)

            x, y = position

            for cx in range(self.width):
                for cy in range(self.height):
                    distance = (
                        abs(cx - x) + abs(cy - y)
                    )

                    if distance == 0:
                        self.belief[cx, cy] *= weight
                    else:
                        self.belief[cx, cy] *= (
                            1 + (weight - 1) / (distance + 1)
                        )

            self.clue_evidence[str(position)] = clue

        self._normalise()

    def _neighbors(self, cell):
        x, y = cell

        for nxt in [
            (x + 1, y), (x - 1, y),
            (x, y + 1), (x, y - 1)
        ]:
            if (
                0 <= nxt[0] < self.width
                and 0 <= nxt[1] < self.height
                and nxt not in self.known_obstacles
            ):
                yield nxt

    def _path_to(self, goal):
        queue = deque([self.position])
        parent = {self.position: None}

        while queue:
            cell = queue.popleft()

            if cell == goal:
                break

            for nxt in self._neighbors(cell):
                if nxt not in parent:
                    parent[nxt] = cell
                    queue.append(nxt)

        if goal not in parent:
            return []

        path = []
        cell = goal

        while cell != self.position:
            path.append(cell)
            cell = parent[cell]

        return list(reversed(path))

    def _safe_to_continue(self, next_cost):
        distance_home = (
            abs(self.position[0] - self.start[0])
            + abs(self.position[1] - self.start[1])
        )

        return (
            self.battery
            >= next_cost + distance_home * self.move_cost
            and self.time_left >= 1 + distance_home
        )

    def choose_action(self):
        if (
            self.position == self.start
            and self.battery < self.max_battery * 0.55
            and self.time_left > 3
        ):
            return {
                "type": "recharge",
                "reason": "Recharge at base because battery is low."
            }

        if self.position not in self.scanned:
            if (
                self.battery >= self.scan_cost
                and self._safe_to_continue(self.scan_cost)
            ):
                return {
                    "type": "scan",
                    "reason": "Scan this cell for clues and signal evidence."
                }

        candidates = []

        for x in range(self.width):
            for y in range(self.height):
                cell = (x, y)

                if (
                    cell == self.position
                    or cell in self.visited
                    or cell in self.known_obstacles
                ):
                    continue

                path = self._path_to(cell)

                if not path:
                    continue

                travel_cost = len(path) * self.move_cost
                home_cost = (
                    abs(x - self.start[0])
                    + abs(y - self.start[1])
                ) * self.move_cost

                if self.battery < travel_cost + home_cost:
                    continue

                score = self.belief[cell] / (
                    1 + 0.12 * len(path)
                )

                candidates.append((score, cell, path))

        if candidates:
            _, target, path = max(
                candidates, key=lambda item: item[0]
            )

            return {
                "type": "move",
                "target": list(path[0]),
                "reason": (
                    f"Move toward promising cell {target}; "
                    f"belief={self.belief[target]:.3f}."
                )
            }

        if self.position == self.start:
            return {
                "type": "return",
                "reason": "No affordable unexplored target remains."
            }

        path = self._path_to(self.start)

        if (
            path
            and self.battery >= len(path) * self.move_cost
            and self.time_left >= len(path)
        ):
            return {
                "type": "move",
                "target": list(path[0]),
                "reason": "Return toward base before resources run out."
            }

        return {
            "type": "return",
            "reason": "Stop exploration because resources are insufficient."
        }

    def observe_and_log(self, action, observation):
        if observation.get("event") == "blocked_by_obstacle":
            target = tuple(action.get("target", []))
            if len(target) == 2:
                self.known_obstacles.add(target)

        self.position = tuple(
            observation.get(
                "position_after",
                observation.get("position", self.position)
            )
        )

        if observation.get("event") in (
            "moved", "EVE_located"
        ):
            self.visited.add(self.position)

        if observation.get("event") in (
            "scan_complete", "EVE_located"
        ) and action["type"] == "scan":
            self.scanned.add(self.position)

        self.battery = observation.get(
            "battery", self.battery
        )
        self.time_left = observation.get(
            "time_left", self.time_left
        )

        self.update_belief(observation)

        best = self.best_estimate()

        self.trace.append({
            "step": len(self.trace) + 1,
            "action": action,
            "observation": observation,
            "reasoning": action.get("reason", ""),
            "top_belief_cell": best["location"],
            "top_belief_probability": best["confidence"]
        })

    def best_estimate(self):
        cell = np.unravel_index(
            np.argmax(self.belief),
            self.belief.shape
        )

        return {
            "location": [int(cell[0]), int(cell[1])],
            "confidence": round(
                float(self.belief[cell]), 4
            )
        }