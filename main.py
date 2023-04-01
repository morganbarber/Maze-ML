import numpy as np
import pygame
import csv
from collections import defaultdict

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.generate_maze()

    def generate_maze(self):
        while True:
            self.maze = np.random.choice([0, 1], (self.height, self.width), p=[0.8, 0.2])
            self.add_border()
            self.ensure_adjacency()
            if self.is_connected():
                break

    def add_border(self):
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        target_index = np.random.randint(1, self.width - 1)
        self.maze[0, target_index] = 2

    def ensure_adjacency(self):
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                if self.maze[y, x] == 1:
                    if self.maze[y - 1, x] == self.maze[y + 1, x] == self.maze[y, x - 1] == self.maze[y, x + 1] == 0:
                        direction = np.random.choice(['up', 'down', 'left', 'right'])
                        if direction == 'up':
                            self.maze[y - 1, x] = 1
                        elif direction == 'down':
                            self.maze[y + 1, x] = 1
                        elif direction == 'left':
                            self.maze[y, x - 1] = 1
                        elif direction == 'right':
                            self.maze[y, x + 1] = 1

    def is_connected(self):
        start, goal = self.get_start_goal_positions()
        stack = [start]
        visited = set()

        while stack:
            current = stack.pop()
            if current == goal:
                return True
            if current in visited:
                continue
            visited.add(current)
            neighbors = self.get_neighbors(current)
            stack.extend(neighbors)

        return False

    def get_neighbors(self, position):
        neighbors = [(position[0] - 1, position[1]), (position[0] + 1, position[1]),
                     (position[0], position[1] - 1), (position[0], position[1] + 1)]
        valid_neighbors = [neighbor for neighbor in neighbors if self.is_valid_position(neighbor)]
        return valid_neighbors

    def is_valid_position(self, position):
        if (0 <= position[0] < self.height) and (0 <= position[1] < self.width):
            if self.maze[position] == 0 or self.maze[position] == 2:
                return True
        return False

    def get_start_goal_positions(self):
        start = np.random.randint(1, self.height - 1), np.random.randint(1, self.width - 1)
        goal = 0, np.where(self.maze[0] == 2)[0][0]
        return start, goal

class Agent:
    def __init__(self, maze, start, goal):
        self.maze = maze
        self.position = start
        self.goal = goal
        self.q_table = defaultdict(lambda: defaultdict(int))
        self.visible = True

    def choose_action(self, state, actions, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(actions)
        q_values = [self.q_table[state][action] for action in actions]
        max_q = max(q_values)
        return actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state, actions, alpha, gamma):
        max_q_next = max([self.q_table[next_state][a] for a in actions])
        self.q_table[state][action] += alpha * (reward + gamma * max_q_next - self.q_table[state][action])

    def move(self, action):
        if action == 0:  # Up
            new_position = (self.position[0] - 1, self.position[1])
        elif action == 1:  # Down
            new_position = (self.position[0] + 1, self.position[1])
        elif action == 2:  # Left
            new_position = (self.position[0], self.position[1] - 1)
        elif action == 3:  # Right
            new_position = (self.position[0], self.position[1] + 1)
        if 0 <= new_position[0] < self.maze.height and 0 <= new_position[1] < self.maze.width:
            if self.maze.maze[new_position] == 0:
                self.position = new_position

def distance(agent, goal):
    return np.sqrt((agent.position[0] - goal[0]) ** 2 + (agent.position[1] - goal[1]) ** 2)

def is_adjacent_or_diagonal(position, goal):
    adjacent_positions = [(goal[0] - 1, goal[1]), (goal[0] + 1, goal[1]), (goal[0], goal[1] - 1), (goal[0], goal[1] + 1),
                          (goal[0] - 1, goal[1] - 1), (goal[0] - 1, goal[1] + 1), (goal[0] + 1, goal[1] - 1),
                          (goal[0] + 1, goal[1] + 1)]
    return position in adjacent_positions

def main(view_training=False):
    pygame.init()
    maze = Maze(100, 100)
    agent_count = 5
    agent_positions = [maze.get_start_goal_positions() for _ in range(agent_count)]
    agents = [Agent(maze, pos[0], pos[1]) for pos in agent_positions]

    actions = [0, 1, 2, 3]
    epsilon = 0.1
    alpha = 0.5
    gamma = 0.99
    episodes = 5000

    screen = pygame.display.set_mode((maze.width * 10, maze.height * 10))
    clock = pygame.time.Clock()

    with open("training_data.csv", "a", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["episode", "agent", "old_position", "action", "new_position", "reward"])

        for episode in range(episodes):
            for agent in agents:
                agent.position = agent_positions[agents.index(agent)][0]
                agent.visible = True
            while True:
                state_data, next_state_data = [], []
                for agent in agents:
                    if not agent.visible:
                        continue

                    agent_actions = [a for a in actions if agent.position != agent_positions[agents.index(agent)][1]]
                    if not agent_actions:
                        agent_actions = actions
                    action = agent.choose_action(agent.position, agent_actions, epsilon)
                    old_position = agent.position
                    agent.move(action)
                    new_position = agent.position

                    if is_adjacent_or_diagonal(new_position, agent_positions[agents.index(agent)][1]):
                        agent.visible = False

                    state_data.append(old_position)
                    next_state_data.append(new_position)

                    if old_position == agent_positions[agents.index(agent)][1]:
                        agent.update_q_table(old_position, action, 100, new_position, agent_actions, alpha, gamma)
                        reward = 100
                    else:
                        distance_reward = -distance(agent, agent_positions[agents.index(agent)][1])
                        agent.update_q_table(old_position, action, distance_reward, new_position, agent_actions, alpha, gamma)
                        reward = distance_reward

                    csvwriter.writerow([episode, agents.index(agent), old_position, action, new_position, reward])

                if view_training:
                    screen.fill((255, 255, 255))

                    for y in range(maze.height):
                        for x in range(maze.width):
                            if maze.maze[y, x] == 1:
                                pygame.draw.rect(screen, (0, 0, 0), (x * 10, y * 10, 10, 10))
                            elif maze.maze[y, x] == 2:
                                pygame.draw.rect(screen, (0, 255, 0), (x * 10, y * 10, 10, 10))

                    for agent in agents:
                        if agent.visible:
                            pygame.draw.rect(screen, (255, 0, 0), (agent.position[1] * 10, agent.position[0] * 10, 10, 10))

                    pygame.display.flip()
                    clock.tick(60)

                if all(not agent.visible for agent in agents):
                    break

            if all(not agent.visible for agent in agents):
                maze.generate_maze()
                agent_positions = [maze.get_start_goal_positions() for _ in range(agent_count)]

    while not view_training:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((255, 255, 255))

        for y in range(maze.height):
            for x in range(maze.width):
                if maze.maze[y, x] == 1:
                    pygame.draw.rect(screen, (0, 0, 0), (x * 10, y * 10, 10, 10))
                elif maze.maze[y, x] == 2:
                    pygame.draw.rect(screen, (0, 255, 0), (x * 10, y * 10, 10, 10))

        for agent in agents:
            if agent.visible:
                pygame.draw.rect(screen, (255, 0, 0), (agent.position[1] * 10, agent.position[0] * 10, 10, 10))

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main(view_training=True)
