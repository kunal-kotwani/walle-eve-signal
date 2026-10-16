import random


class MissionEnvironment:
    def __init__(self, config):
        self.config = config

        world = config["map"]
        self.width = world["width"]
        self.height = world["height"]
        self.start = tuple(config["initial_position"])
        self.position = self.start

        resources = config["resources"]
        self.battery = resources["battery"]
        self.max_battery = resources["max_battery"]
        self.time_left = resources["time_budget"]

        self.move_cost = resources["move_cost"]
        self.scan_cost = resources["scan_cost"]
        self.recharge_cost = resources["recharge_cost"]
        self.return_cost = resources["return_cost"]

        self.done = False
        self.eve_found = False
        self.returned_to_base = False

        rng = random.Random(world["seed"])

        cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) != self.start
        ]

        self.eve = rng.choice(cells)
        cells.remove(self.eve)

        obstacle_count = min(
            world["obstacle_count"], len(cells)
        )
        self.obstacles = set(
            rng.sample(cells, obstacle_count)
        )

        clue_rng = random.Random(
            config["clues"]["seed"]
        )

        clue_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.obstacles
            and (x, y) != self.start
        ]

        clue_count = min(
            config["clues"]["count"], len(clue_cells)
        )

        clue_types = [
            "metal_trace",
            "power_residue",
            "EVE_signal_fragment"
        ]

        self.clues = {
            cell: clue_rng.choice(clue_types)
            for cell in clue_rng.sample(
                clue_cells, clue_count
            )
        }

        self.visited = {self.start}
        self.scanned = set()

    def in_bounds(self, cell):
        x, y = cell
        return (
            0 <= x < self.width
            and 0 <= y < self.height
        )

    def signal_strength(self, cell):
        distance = (
            abs(cell[0] - self.eve[0])
            + abs(cell[1] - self.eve[1])
        )

        rng = random.Random(
            self.config["map"]["seed"] * 100000
            + cell[0] * 101
            + cell[1] * 17
        )

        noise = rng.uniform(-4, 4)
        return round(
            max(0, min(100, 100 - 12 * distance + noise)),
            2
        )

    def initial_observation(self):
        return {
            "event": "mission_started",
            "position": list(self.position),
            "signal_strength":
                self.config["initial_signal"]["strength"],
            "battery": self.battery,
            "time_left": self.time_left
        }

    def step(self, action):
        if self.done:
            return {"event": "mission_already_ended"}

        kind = action["type"]

        observation = {
            "action": kind,
            "position_before": list(self.position)
        }

        if kind == "move":
            target = tuple(action["target"])

            if not self.in_bounds(target):
                event = "invalid_target"

            elif target in self.obstacles:
                event = "blocked_by_obstacle"

            elif (
                abs(target[0] - self.position[0])
                + abs(target[1] - self.position[1])
                != 1
            ):
                event = "move_must_be_adjacent"

            elif (
                self.battery < self.move_cost
                or self.time_left < 1
            ):
                event = "insufficient_resources"
                self.done = True

            else:
                self.battery -= self.move_cost
                self.time_left -= 1
                self.position = target
                self.visited.add(target)

                event = "moved"
                observation["signal_strength"] = (
                    self.signal_strength(self.position)
                )

                if self.position == self.eve:
                    self.eve_found = True
                    self.done = True
                    event = "EVE_located"

            observation["event"] = event

        elif kind == "scan":
            if (
                self.battery < self.scan_cost
                or self.time_left < 1
            ):
                observation["event"] = (
                    "insufficient_resources"
                )
                self.done = True

            else:
                self.battery -= self.scan_cost
                self.time_left -= 1
                self.scanned.add(self.position)

                clue = self.clues.get(
                    self.position, "none"
                )

                observation.update({
                    "event": "scan_complete",
                    "signal_strength":
                        self.signal_strength(self.position),
                    "clue": clue,
                    "clue_found": clue != "none"
                })

                if self.position == self.eve:
                    self.eve_found = True
                    self.done = True
                    observation["event"] = "EVE_located"

        elif kind == "recharge":
            if self.position != self.start:
                observation["event"] = (
                    "recharge_only_at_base"
                )

            elif self.time_left < self.recharge_cost:
                observation["event"] = (
                    "not_enough_time_to_recharge"
                )
                self.done = True

            else:
                self.battery = self.max_battery
                self.time_left -= self.recharge_cost
                observation["event"] = "recharged_at_base"

        elif kind == "return":
            if self.position == self.start:
                self.returned_to_base = True
                self.done = True
                observation["event"] = "returned_to_base"
            else:
                observation["event"] = (
                    "return_action_requires_base"
                )

        else:
            observation["event"] = "unknown_action"

        observation.update({
            "position_after": list(self.position),
            "battery": self.battery,
            "time_left": self.time_left,
            "done": self.done
        })

        return observation