# Maze-ML (Q-Learning)

A maze solving script using Q-learning algorithm to train agents to navigate through randomly generated mazes.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Editing](#editing)
- [Credits](#credits)

## Features

- Random maze generation
- Training multiple agents simultaneously
- Saving training data in a CSV file
- Visualization of the agents navigating through the maze

## Installation

1. Clone the repository
```bash
git clone https://github.com/morganbarber/maze-ML.git
```

2. Change the current directory to the cloned repository
```bash
cd maze-ML
```

3. Install the required packages
```bash
pip install numpy pygame
```

## Usage

1. Run the script
```bash
python main.py
```

2. The script will start training the agents and visualize their progress in a separate Pygame window.

3. The training data will be saved in `training_data.csv`

## Editing

To edit the script, open `main.py` in your favorite text editor or Python IDE.

Here are some of the parameters that you can modify:

- `maze = Maze(100, 100)`: Change the maze dimensions (width, height)
- `agent_count = 5`: Change the number of agents being trained simultaneously
- `epsilon = 0.1`: Change the exploration rate of the agents
- `alpha = 0.5`: Change the learning rate of the agents
- `gamma = 0.99`: Change the discount factor of the agents
- `episodes = 5000`: Change the number of episodes for training

## Credits

This script is developed by [Morgan Barber](https://github.com/morganbarber).

Read Me file generated by GPT-4.
