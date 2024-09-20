import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time


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


# Custom environment class for Robot Sumo
class RobotSumoEnv(gym.Env):
    def __init__(self):
        super(RobotSumoEnv, self).__init__()

        # Define the action and observation space
        self.action_space = spaces.Discrete(4)  # Move forward, turn left, turn right, move backward
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)

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

        return observation, 0, False, {}, {}

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


# Main function to test the environment
if __name__ == "__main__":
    env = RobotSumoEnv()

    # Loop for 10 matches
    for match in range(10):
        print(f"Starting match {match + 1}")
        cumulative_score = 0  # Track the cumulative score for each match
        max_score = 1  # The maximum possible score for each match (if robot wins)

        observation, info = env.reset()

        # Each match will run for 50 steps
        for step in range(50):
            # Get the robot's action based on its decision strategy
            robot_action = robot_decision(env.robot_position, env.robot_orientation, env.opponent_position)

            # Get the opponent's action based on its decision strategy
            opponent_action = opponent_decision(env.opponent_position, env.opponent_orientation, env.robot_position)

            # Simulate a step for both robots
            observation, reward, done, _, info = env.step(robot_action)

            cumulative_score += reward  # Add the reward from each step to the cumulative score

            # Handle Pygame events to keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    exit()

            # Render the current state of the environment
            env.render()

            # Slow down the simulation (optional)
            time.sleep(0.2)

            if done:
                print(f"Match {match + 1} over at step {step + 1}. Resetting environment.")
                break  # Break the step loop when the match is over

        print(f"Match {match + 1} score: {cumulative_score}/{max_score}")

    env.close()
