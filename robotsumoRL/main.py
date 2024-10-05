import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time
import tensorflow as tf
from tensorflow.keras import layers


# Helper function to calculate the angle to opponent
def calculate_angle_to_opponent(robot_position, opponent_position):
    direction_vector = opponent_position - robot_position
    angle_to_opponent = np.arctan2(direction_vector[1], direction_vector[0])
    return angle_to_opponent


# Decision function for the robot to flank and attack the opponent
def robot_decision(robot_position, robot_orientation, opponent_position):
    angle_to_opponent = calculate_angle_to_opponent(robot_position, opponent_position)

    # Calculate the angle difference between robot's orientation and opponent's position
    angle_diff = angle_to_opponent - robot_orientation

    # Normalize the angle difference to the range [-π, π]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    if abs(angle_diff) < 0.1:  # Robot is facing the opponent
        return 0  # Move forward to attack
    elif angle_diff > 0:  # Opponent is to the right
        return 2  # Turn right to face the opponent
    else:  # Opponent is to the left
        return 1  # Turn left to face the opponent


# Decision function for the opponent to flank and attack the robot
def opponent_decision(opponent_position, opponent_orientation, robot_position):
    angle_to_robot = calculate_angle_to_opponent(opponent_position, robot_position)

    # Calculate the angle difference between opponent's orientation and robot's position
    angle_diff = angle_to_robot - opponent_orientation

    # Normalize the angle difference to the range [-π, π]
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

    if abs(angle_diff) < 0.1:  # Opponent is facing the robot
        return 0  # Move forward to attack
    elif angle_diff > 0:  # Robot is to the right
        return 2  # Turn right to face the robot
    else:  # Robot is to the left
        return 1  # Turn left to face the robot


# DQN Agent Class
class DQNAgent:
    def __init__(self, model):
        self.model = model
        self.gamma = 0.95  # Discount rate for future rewards
        self.epsilon = 1.0  # Exploration rate (start with full exploration)
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for epsilon
        self.memory = []  # Replay memory for storing past experiences
        self.batch_size = 32  # Size of training batches

    def choose_action(self, state):
        # Exploration-exploitation tradeoff
        if np.random.rand() <= self.epsilon:
            return np.random.choice(env.action_space.n)  # Choose random action
        # Predict action based on current state
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])  # Return action with max Q-value

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return  # Don't train until we have enough samples

        # Sample a batch of experiences from memory
        minibatch = np.random.choice(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Update Q-value using Bellman equation
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, :], verbose=0)[0])

            # Update the Q-values for the selected action
            q_values = self.model.predict(state[np.newaxis, :], verbose=0)
            q_values[0][action] = target
            self.model.fit(state[np.newaxis, :], q_values, verbose=0)

        # Decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Create the model for the agent
def create_model():
    model = tf.keras.Sequential()
    model.add(layers.InputLayer(input_shape=(6,)))  # The input shape is based on your observation space
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(env.action_space.n, activation='linear'))  # Output size should match action space
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# Custom environment class for Robot Sumo
class RobotSumoEnv(gym.Env):
    def __init__(self):
        super(RobotSumoEnv, self).__init__()

        # Define the action and observation space
        self.action_space = spaces.Discrete(4)  # Move forward, turn left, turn right, move backward
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

        # Initialize positions of robot and opponent
        self.robot_position = np.array([0.5, 0.5])
        self.opponent_position = np.array([0.7, 0.7])
        self.ring_radius = 1.0

        # Initialize orientations of robot and opponent
        self.robot_orientation = 0.0  # Facing upward initially (angle in radians)
        self.opponent_orientation = np.pi  # Facing downward initially

        # Initialize Pygame for rendering
        pygame.init()
        self.window_size = 600
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def reset(self):
        self.robot_position = np.array([0.5, 0.5])
        self.opponent_position = np.array([0.7, 0.7])
        self.robot_orientation = 0.0
        self.opponent_orientation = np.pi
        observation = np.concatenate(
            [self.robot_position, [self.robot_orientation], self.opponent_position, [self.opponent_orientation]])
        return observation, {}

    def step(self, action):
        # Robot's movement based on orientation
        if action == 0:  # Move forward
            self.robot_position[0] += 0.1 * np.cos(self.robot_orientation)
            self.robot_position[1] += 0.1 * np.sin(self.robot_orientation)
        elif action == 1:  # Turn left
            self.robot_orientation -= 0.1
        elif action == 2:  # Turn right
            self.robot_orientation += 0.1
        elif action == 3:  # Move backward
            self.robot_position[0] -= 0.1 * np.cos(self.robot_orientation)
            self.robot_position[1] -= 0.1 * np.sin(self.robot_orientation)

        # Opponent's AI: Move toward the robot
        opponent_direction = (self.robot_position - self.opponent_position)
        opponent_orientation_diff = np.arctan2(opponent_direction[1], opponent_direction[0]) - self.opponent_orientation

        if opponent_orientation_diff > 0:
            self.opponent_orientation += 0.05
        else:
            self.opponent_orientation -= 0.05

        self.opponent_position[0] += 0.05 * np.cos(self.opponent_orientation)
        self.opponent_position[1] += 0.05 * np.sin(self.opponent_orientation)

        # Observation includes positions and orientations
        observation = np.concatenate(
            [self.robot_position, [self.robot_orientation], self.opponent_position, [self.opponent_orientation]])

        # Reward based on position relative to opponent (you can improve this logic)
        reward = -np.linalg.norm(self.robot_position - self.opponent_position)
        done = False  # Modify this when you want to signal the end of an episode (e.g., if someone is pushed out)

        return observation, reward, done, {}, {}

    def render(self, mode='human'):
        # Convert the robot and opponent positions to Pygame coordinates
        def to_pygame_coords(position):
            return int(self.window_size * position[0]), int(self.window_size * position[1])

        # Fill the screen with white
        self.screen.fill((255, 255, 255))

        # Draw the sumo ring
        pygame.draw.circle(self.screen, (0, 0, 0), (self.window_size // 2, self.window_size // 2),
                           int(self.ring_radius * self.window_size / 2), 2)

        # Draw the robot (blue) and opponent (red)
        robot_coords = to_pygame_coords(self.robot_position)
        opponent_coords = to_pygame_coords(self.opponent_position)
        pygame.draw.circle(self.screen, (0, 0, 255), robot_coords, 20)
        pygame.draw.circle(self.screen, (255, 0, 0), opponent_coords, 20)

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        self.clock.tick(10)

    def close(self):
        pygame.quit()


# Main function to test the environment and train the DQN agent
if __name__ == "__main__":
    env = RobotSumoEnv()
    model = create_model()
    agent = DQNAgent(model)

    episodes = 1000  # Number of episodes to train the agent
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.array(state)
        total_reward = 0

        for step in range(50):
            # Get the robot's action from the agent
            action = agent.choose_action(state)

            # Take the action in the environment
            next_state, reward, done, _, info = env.step(action)
            next_state = np.array(next_state)

            # Store the experience in the agent's memory
            agent.remember(state, action, reward, next_state, done)

            # Train the agent
            agent.replay()

            state = next_state
            total_reward += reward

            # Handle Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()

            # Render the current state of the environment
            env.render()

            if done:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}")
                break

        # Optionally slow down the simulation (remove for faster training)
        time.sleep(0.1)

    # Save the model after training
    agent.model.save('robot_sumo_dqn_model.h5')

    env.close()
