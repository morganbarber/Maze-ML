# Maze-ML (Q-Learning)

A maze solving script using Q-learning algorithm to train agents to navigate through randomly generated mazes.

Note: The training data CSV file can be hefty on your devices storage.

I will turn this into a deep learning model soon.

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

# Paper

Title: Reinforcement Learning-based Pathfinding in Randomly Generated Mazes

Abstract: 

In this paper, we present a pathfinding algorithm using reinforcement learning in randomly generated mazes. The algorithm utilizes multiple agents and Q-learning to find the optimal path to the goal location. The agents are trained using different maze configurations, and their performance is evaluated based on the distance from the goal. The simulations are visualized using the Pygame library. The results show that the agents can successfully navigate through the mazes and find the shortest paths to their target locations.

1. Introduction

Pathfinding is an important aspect of many applications, including robotics, video games, and transportation systems. Traditional pathfinding algorithms, such as Dijkstra's and A* algorithms, provide optimal solutions in known environments. However, in complex and dynamic environments where the agents have no prior knowledge of the environment, reinforcement learning can be used to learn the optimal path online.

In this paper, we present a reinforcement learning-based pathfinding algorithm that uses Q-learning to find the optimal path in randomly generated mazes. The algorithm is implemented using Python and the Pygame library for visualization. The agents are trained on different maze configurations, and their performance is evaluated based on the distance from the goal. The algorithm is robust and can handle varying maze sizes and complexities.

2. Methodology

2.1 Maze Generation

The mazes are generated using NumPy's random choice function, which creates a random matrix of 0s (empty spaces) and 1s (walls) with specified probabilities. The mazes are ensured to be connected and have a border surrounding the entire maze. The starting and goal positions are randomly selected within the maze.

2.2 Agent and Q-learning

The agents are initialized with a Q-table that maps states (positions in the maze) to actions (moving up, down, left, or right). The agents choose actions based on the epsilon-greedy strategy, where they either choose a random action or the action with the highest Q-value for the current state. The Q-values are updated using the Q-learning update rule with a learning rate (alpha) and a discount factor (gamma).

The agents receive a reward based on the distance from the goal position. A positive reward is given when the agent reaches the goal, while a negative reward proportional to the distance from the goal is given otherwise. The agents are trained over multiple episodes, with each episode consisting of the agent navigating through a randomly generated maze.

2.3 Simulation and Visualization

The simulation is carried out using the Pygame library, which provides a visualization of the maze and the agents' positions. The agents can be set to be visible or invisible, depending on whether their training progress is to be observed. The training data, including the agents' positions, actions, and rewards, are recorded in a CSV file for further analysis.

3. Results

The agents successfully learn to navigate through the mazes and find the shortest paths to their target locations. The algorithm is robust and can handle varying maze sizes and complexities. The performance of the agents improves over time as they learn from their experiences in different maze configurations.

4. Conclusion

In conclusion, we have presented a reinforcement learning-based pathfinding algorithm that utilizes Q-learning to find the optimal path in randomly generated mazes. The agents are trained on different maze configurations, and their performance is evaluated based on the distance from the goal. The algorithm is robust and can handle varying maze sizes and complexities. The simulations are visualized using the Pygame library, which provides a useful tool for observing the agents' learning progress. Future work can involve improving the reward structure, incorporating additional state information, and exploring other reinforcement learning algorithms for pathfinding.
