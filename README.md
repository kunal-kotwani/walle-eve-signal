# WALL-E and EVE's Signal

An autonomous search simulation in which WALL-E explores a hidden grid,
gathers clues, updates its belief about EVE's location, and manages
limited battery and time resources.

## Features

- Configurable grid world using JSON
- Hidden EVE location, obstacles and clues
- Signal-based probabilistic belief updates
- Autonomous movement and scanning
- Battery and time management
- Action trace with observations and reasoning
- JSON and text mission reports
- Map visualization

## Technology

- Python 3
- NumPy
- Matplotlib
- JSON

## Setup

1. Install Python 3.
2. Open this folder in VS Code.
3. Select a Python interpreter.
4. Install the packages listed in requirements.txt.
5. Run main.py using the VS Code Run Python File button.

## Configuration

The sample mission is defined in:

configs/sample_mission.json

It includes map dimensions, map seed, initial position,
signal reading, resource budget, clue settings and agent settings.

## Outputs

Running the mission generates:

- outputs/action_trace.jsonl
- outputs/mission_report.json
- outputs/mission_report.txt
- outputs/mission_map.png

## Architecture

1. Environment creates a hidden world.
2. Agent receives observations through actions.
3. Belief system updates possible EVE locations.
4. Planner selects moves and scans under resource constraints.
5. Logger records actions, observations and reasoning.
6. Reporter saves the final mission outcome.

## Limitations

This version uses probabilistic inference and rule-based planning.
It does not yet train a machine-learning model. Its signal model
is simplified, and the reported confidence is not calibrated
against real-world observations.

## Future Improvements

- A* pathfinding
- Frontier-based exploration
- Evaluation across multiple random seeds
- Comparison with a random-search baseline
- Q-learning or another reinforcement-learning approach
- Interactive dashboard