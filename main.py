import argparse
import json
from pathlib import Path

from environment import MissionEnvironment
from agent import WallEAgent
from visualize import save_map


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


def run_mission(config_path):
    config_path = Path(config_path)

    root = Path(__file__).resolve().parent
    output_dir = root / "outputs"
    output_dir.mkdir(exist_ok=True)

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    env = MissionEnvironment(config)
    agent = WallEAgent(config)

    initial = env.initial_observation()
    agent.update_belief(initial)

    agent.trace.append({
        "step": 0,
        "action": {"type": "start"},
        "observation": initial,
        "reasoning": "Initialize the belief from the starting signal.",
        "top_belief_cell": agent.best_estimate()["location"],
        "top_belief_probability":
            agent.best_estimate()["confidence"]
    })

    max_steps = config["agent"]["max_steps"]

    for _ in range(max_steps):
        if env.done:
            break

        action = agent.choose_action()
        observation = env.step(action)

        agent.observe_and_log(action, observation)

    if not env.done:
        ending = "step_limit_reached"
    elif env.eve_found:
        ending = "eve_located"
    elif env.returned_to_base:
        ending = "returned_to_base"
    else:
        ending = "resources_exhausted_or_stopped"

    estimate = agent.best_estimate()

    report = {
        "mission_name": config["mission_name"],
        "ending": ending,
        "eve_located": env.eve_found,
        "evidence_found": agent.clue_evidence,
        "inferred_eve_location": estimate["location"],
        "confidence": estimate["confidence"],
        "actions_taken": len(agent.trace) - 1,
        "visited_cells": [
            list(cell) for cell in sorted(env.visited)
        ],
        "scanned_cells": [
            list(cell) for cell in sorted(env.scanned)
        ],
        "remaining_battery": env.battery,
        "remaining_time": env.time_left,
        "recommended_next_step": (
            "Send a confirmation or recovery team to EVE."
            if env.eve_found
            else "Review evidence and launch another mission."
        )
    }

    write_jsonl(
        output_dir / "action_trace.jsonl",
        agent.trace
    )

    with (output_dir / "mission_report.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    with (output_dir / "mission_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(
            f"MISSION REPORT: {report['mission_name']}\n"
        )
        file.write("=" * 40 + "\n")

        for key, value in report.items():
            file.write(f"{key}: {value}\n")

    save_map(
        env, agent,
        output_dir / "mission_map.png"
    )

    print(json.dumps(report, indent=2))
    print("\nMission finished.")
    print(f"Trace: {output_dir / 'action_trace.jsonl'}")
    print(f"Report: {output_dir / 'mission_report.txt'}")
    print(f"Map: {output_dir / 'mission_map.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sample_mission.json"
    )
    args = parser.parse_args()

    run_mission(args.config)